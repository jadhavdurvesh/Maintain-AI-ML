from __future__ import annotations

import os
from functools import lru_cache

MODEL_ID = os.getenv("MAINTAIN_CHRONOS_MODEL", "amazon/chronos-t5-tiny")


def _is_chronos2() -> bool:
    return MODEL_ID.startswith("amazon/chronos-2") or "chronos-2" in MODEL_ID


@lru_cache(maxsize=1)
def get_pipeline():
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Free Render instances have very limited CPU/RAM. Keep inference bounded.
    if device == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    if _is_chronos2():
        from chronos import Chronos2Pipeline

        return Chronos2Pipeline.from_pretrained(
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


def status() -> dict:
    try:
        import chronos  # noqa: F401
        import torch

        return {
            "available": True,
            "model": MODEL_ID,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "family": "chronos-2" if _is_chronos2() else "chronos-t5",
        }
    except Exception as exc:
        return {
            "available": False,
            "model": MODEL_ID,
            "reason": str(exc),
            "family": "chronos-2" if _is_chronos2() else "chronos-t5",
        }


def forecast(values: list[float], horizon: int) -> list[float]:
    import torch

    pipeline = get_pipeline()
    context = torch.tensor(values[-128:], dtype=torch.float32)

    # One trajectory is enough for the demo point forecast and is substantially
    # cheaper than the Chronos default sampling count on a free CPU instance.
    with torch.inference_mode():
        if _is_chronos2():
            predictions = pipeline.predict(context, prediction_length=horizon)
        else:
            predictions = pipeline.predict(
                context,
                prediction_length=horizon,
                num_samples=1,
            )

    tensor = predictions[0].detach().float().cpu()

    if _is_chronos2():
        if tensor.ndim == 3:
            tensor = tensor[0]
        if tensor.ndim == 2:
            quantile_levels = getattr(pipeline, "quantiles", None)
            if quantile_levels is not None and 0.5 in quantile_levels:
                median_index = list(quantile_levels).index(0.5)
            else:
                median_index = tensor.shape[-1] // 2
            return tensor[:, median_index].tolist()[:horizon]
    else:
        if tensor.ndim == 2:
            return tensor.median(dim=0).values.tolist()[:horizon]

    return tensor.reshape(-1).tolist()[:horizon]
