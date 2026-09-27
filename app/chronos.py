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
        return {"available": True, "model": MODEL_ID, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    except Exception as exc:
        return {"available": False, "model": MODEL_ID, "reason": str(exc)}


def forecast(values: list[float], horizon: int) -> list[float]:
    import torch
    pipeline = get_pipeline()
    context = torch.tensor(values[-512:], dtype=torch.float32)
    output = pipeline.predict(context, prediction_length=horizon)
    tensor = output.detach().float().cpu()
    # Chronos returns samples x batch x horizon or batch x samples x horizon depending on version.
    if tensor.ndim == 3:
        batch = tensor[0]
        median = batch.median(dim=0).values if batch.shape[0] > 1 else batch[0]
        return median.tolist()
    return tensor.reshape(-1).tolist()[:horizon]
