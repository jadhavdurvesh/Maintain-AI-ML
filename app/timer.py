from __future__ import annotations

import os
from functools import lru_cache

MODEL_ID = os.getenv("MAINTAIN_TIMER_MODEL", "thuml/timer-base-84m")


def _context_length() -> int:
    try:
        # Timer supports much longer contexts, but a small context dramatically
        # reduces CPU RAM/activation pressure on the free Render instance.
        return max(96, min(2880, int(os.getenv("MAINTAIN_TIMER_CONTEXT", "576"))))
    except ValueError:
        return 576


def _dtype(torch):
    requested = os.getenv("MAINTAIN_TIMER_DTYPE", "fp16").strip().lower()
    if requested in {"fp32", "float32", "32"}:
        return torch.float32
    if requested in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float16


@lru_cache(maxsize=1)
def get_model():
    import torch
    from transformers import AutoModelForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    # Chronos and Timer must not live in RAM together on the free instance.
    try:
        from .chronos import release_pipeline
        release_pipeline()
    except Exception:
        pass

    dtype = _dtype(torch) if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=dtype,
    )
    model = model.to(device).eval()

    # KV caching is useful for large autoregressive text generation, but Timer
    # forecasts are short and the cache costs avoidable memory on this service.
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    return model, device


def release_model() -> None:
    """Unload Timer and return its memory to the process/OS where possible."""
    get_model.cache_clear()
    try:
        import gc
        gc.collect()
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def status() -> dict:
    enabled = os.getenv("MAINTAIN_ENABLE_TIMER", "0").strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        return {"available": False, "model": "Timer", "checkpoint": MODEL_ID, "reason": "Timer is disabled by configuration."}
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
        return {
            "available": True,
            "model": "Timer",
            "checkpoint": MODEL_ID,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "context": _context_length(),
            "dtype": os.getenv("MAINTAIN_TIMER_DTYPE", "fp16"),
            "memory_mode": "lazy + exclusive",
        }
    except Exception as exc:
        return {"available": False, "model": "Timer", "checkpoint": MODEL_ID, "reason": str(exc)}


def forecast(values: list[float], horizon: int) -> list[float]:
    if len(values) < 32:
        raise ValueError("Timer requires at least 32 sensor samples")
    if not status().get("available"):
        raise RuntimeError(status().get("reason", "Timer is unavailable"))
    if horizon < 1 or horizon > 64:
        raise ValueError("Timer horizon must be between 1 and 64")

    import numpy as np
    import torch

    x = np.asarray(values[-_context_length():], dtype=np.float32)
    mean = float(x.mean())
    std = float(x.std() or 1.0)
    sequence = torch.tensor(((x - mean) / std)[None], dtype=_dtype(torch))
    model, device = get_model()

    try:
        with torch.inference_mode():
            output = model.generate(sequence.to(device), max_new_tokens=horizon)
        prediction = output[0, -horizon:].detach().float().cpu().numpy() * std + mean
        return prediction.tolist()
    finally:
        # Timer is explicitly an on-demand secondary model. Releasing it after
        # every request prevents it from accumulating alongside Chronos.
        release_model()
