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
    # Keep CPU inference deliberately lightweight for free-tier deployments.
    if device == "cpu":
        torch.set_num_threads(max(1, min(2, os.cpu_count() or 1)))

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
    # The free-tier model only needs the recent context. Keeping this bounded
    # also prevents accidental memory growth if a caller sends a long series.
    context = torch.tensor(values[-512:], dtype=torch.float32)
    predictions = pipeline.predict(context, prediction_length=horizon)

    tensor = predictions[0].detach().float().cpu()

    if _is_chronos2():
        # Chronos-2 returns [batch, horizon, quantiles]. Select the 0.5 quantile.
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
        # Classic Chronos returns [batch, samples, horizon]. Use the median
        # across sampled trajectories for a stable point forecast.
        if tensor.ndim == 2:
            return tensor.median(dim=0).values.tolist()[:horizon]

    return tensor.reshape(-1).tolist()[:horizon]
