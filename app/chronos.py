from __future__ import annotations

import os
from functools import lru_cache

MODEL_ID = os.getenv("MAINTAIN_CHRONOS_MODEL", "amazon/chronos-2")


@lru_cache(maxsize=1)
def get_pipeline():
    from chronos import Chronos2Pipeline
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    return Chronos2Pipeline.from_pretrained(MODEL_ID, device_map=device)


def status() -> dict:
    try:
        import chronos  # noqa: F401
        import torch

        return {
            "available": True,
            "model": MODEL_ID,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
    except Exception as exc:
        return {"available": False, "model": MODEL_ID, "reason": str(exc)}


def forecast(values: list[float], horizon: int) -> list[float]:
    import torch

    pipeline = get_pipeline()
    context = torch.tensor(values[-512:], dtype=torch.float32)
    predictions = pipeline.predict(context, prediction_length=horizon)

    # Chronos-2 returns one tensor per target. For a univariate series the
    # tensor is [1, horizon, quantiles]. The 0.5 quantile is the point forecast.
    tensor = predictions[0].detach().float().cpu()
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim == 2:
        quantile_levels = getattr(pipeline, "quantiles", None)
        if quantile_levels is not None and 0.5 in quantile_levels:
            median_index = list(quantile_levels).index(0.5)
        else:
            median_index = tensor.shape[-1] // 2
        return tensor[:, median_index].tolist()[:horizon]

    return tensor.reshape(-1).tolist()[:horizon]
