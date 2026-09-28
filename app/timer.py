from __future__ import annotations

import os
from functools import lru_cache

MODEL_ID = os.getenv("MAINTAIN_TIMER_MODEL", "thuml/timer-base-84m")


@lru_cache(maxsize=1)
def get_model():
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)
    return model.to(device).eval(), device


def status() -> dict:
    enabled = os.getenv("MAINTAIN_ENABLE_TIMER", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return {"available": False, "model": "Timer", "checkpoint": MODEL_ID, "reason": "Timer is disabled by configuration."}
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return {"available": True, "model": "Timer", "checkpoint": MODEL_ID, "device": "cuda" if torch.cuda.is_available() else "cpu"}
    except Exception as exc:
        return {"available": False, "model": "Timer", "checkpoint": MODEL_ID, "reason": str(exc)}


def forecast(values: list[float], horizon: int) -> list[float]:
    if len(values) < 32:
        raise ValueError("Timer requires at least 32 sensor samples")
    if not status().get("available"):
        raise RuntimeError(status().get("reason", "Timer is unavailable"))

    import numpy as np
    import torch

    x = np.asarray(values[-2880:], dtype=np.float32)
    mean = float(x.mean())
    std = float(x.std() or 1.0)
    sequence = torch.tensor(((x - mean) / std)[None], dtype=torch.float32)
    model, device = get_model()

    with torch.inference_mode():
        output = model.generate(sequence.to(device), max_new_tokens=horizon)
    prediction = output[0, -horizon:].detach().float().cpu().numpy() * std + mean
    return prediction.tolist()
