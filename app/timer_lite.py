"""Memory-safe Timer-Lite fallback for low-memory CPU deployments.

This is intentionally not the official THUML Timer checkpoint. It preserves the
same forecast API contract while using a tiny deterministic trend model so the
free Render instance does not need to load the 84M-parameter Timer checkpoint.
"""
from __future__ import annotations

import numpy as np


def status() -> dict:
    return {
        "available": True,
        "model": "Timer-Lite",
        "device": "cpu",
        "memory_mode": "statistical-lightweight",
        "official_timer": False,
    }


def forecast(values: list[float], horizon: int) -> list[float]:
    if len(values) < 16:
        raise ValueError("Timer-Lite requires at least 16 sensor samples")
    if horizon < 1 or horizon > 64:
        raise ValueError("Timer-Lite horizon must be between 1 and 64")

    x = np.asarray(values[-64:], dtype=np.float64)
    if not np.isfinite(x).all():
        x = x[np.isfinite(x)]
    if len(x) < 16:
        raise ValueError("Timer-Lite requires at least 16 finite sensor samples")

    # Smooth only the noise used to estimate the local trend. The returned
    # forecast stays on the original signal scale.
    window = min(5, max(3, len(x) // 8))
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(x, kernel, mode="same")
    recent = smooth[-min(24, len(smooth)):]
    t = np.arange(len(recent), dtype=np.float64)
    slope, intercept = np.polyfit(t, recent, 1)

    # Dampen the long-range slope so noisy sensor streams do not explode.
    last = float(x[-1])
    scale = max(float(np.std(x)), abs(last) * 0.01, 1e-6)
    slope = float(np.clip(slope, -0.25 * scale, 0.25 * scale))
    residual = float(recent[-1] - last)

    steps = np.arange(1, horizon + 1, dtype=np.float64)
    damp = 1.0 - np.exp(-steps / 8.0)
    forecast = last + residual * np.exp(-steps / 3.0) + slope * steps * damp
    return forecast.astype(float).tolist()
