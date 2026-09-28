from __future__ import annotations

import os
from functools import lru_cache

MODEL_ID = os.getenv("MAINTAIN_CHRONOS_MODEL", "amazon/chronos-bolt-tiny")

# Normalize the friendly names accepted by the API/UI to Hugging Face ids.
MODEL_ALIASES = {
    "chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
    "amazon/chronos-bolt-tiny": "amazon/chronos-bolt-tiny",
    "chronos-t5-tiny": "amazon/chronos-t5-tiny",
    "amazon/chronos-t5-tiny": "amazon/chronos-t5-tiny",
    "chronos-2": "amazon/chronos-2",
    "amazon/chronos-2": "amazon/chronos-2",
}
MODEL_ID = MODEL_ALIASES.get(MODEL_ID, MODEL_ID)


def _family() -> str:
    if "chronos-2" in MODEL_ID:
        return "chronos-2"
    if "bolt" in MODEL_ID:
        return "chronos-bolt"
    return "chronos-t5"


@lru_cache(maxsize=1)
def get_pipeline():
    import torch

    # Timer is the memory-heavy secondary model. Never keep both models resident
    # on the same low-memory Render instance.
    try:
        from .timer import release_model
        release_model()
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    if _family() == "chronos-2":
        from chronos import Chronos2Pipeline
        return Chronos2Pipeline.from_pretrained(
            MODEL_ID,
            device_map=device,
            torch_dtype=torch.float32,
        )

    if _family() == "chronos-bolt":
        from chronos import ChronosBoltPipeline
        return ChronosBoltPipeline.from_pretrained(
            MODEL_ID,
            device_map=device,
            torch_dtype=torch.float32,
        )

    from chronos import ChronosPipeline
    return ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map=device,
        torch_dtype=torch.float32,
    )


def release_pipeline() -> None:
    """Release the cached Chronos model so a secondary model can use the RAM."""
    get_pipeline.cache_clear()
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def status() -> dict:
    try:
        import chronos  # noqa: F401
        import torch
        return {
            "available": True,
            "model": MODEL_ID,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "family": _family(),
        }
    except Exception as exc:
        return {
            "available": False,
            "model": MODEL_ID,
            "reason": str(exc),
            "family": _family(),
        }


def forecast(values: list[float], horizon: int) -> list[float]:
    import torch

    if len(values) < 32:
        raise ValueError("Chronos requires at least 32 sensor samples")

    pipeline = get_pipeline()
    # Keep the context bounded for free CPU instances.
    context = torch.tensor(values[-128:], dtype=torch.float32)

    with torch.inference_mode():
        if _family() == "chronos-2":
            predictions = pipeline.predict(context, prediction_length=horizon)
        elif _family() == "chronos-bolt":
            predictions = pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=1,
            )
        else:
            predictions = pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=1,
            )

    tensor = predictions[0].detach().float().cpu()

    # Chronos-Bolt/T5 return sampled or quantile-shaped tensors depending on
    # version. Reduce those shapes to one point forecast for the app.
    if tensor.ndim == 2:
        if _family() == "chronos-2":
            quantile_levels = getattr(pipeline, "quantiles", None)
            if quantile_levels is not None and 0.5 in quantile_levels:
                median_index = list(quantile_levels).index(0.5)
                tensor = tensor[:, median_index]
            else:
                tensor = tensor.median(dim=-1).values
        else:
            tensor = tensor.median(dim=0).values

    return tensor.reshape(-1).tolist()[:horizon]
