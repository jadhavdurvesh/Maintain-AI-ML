from typing import Literal

from pydantic import BaseModel, Field


ChronosModel = Literal[
    "amazon/chronos-bolt-tiny",
    "amazon/chronos-t5-tiny",
    "amazon/chronos-2",
    "chronos-bolt-tiny",
    "chronos-t5-tiny",
    "chronos-2",
    "timer",
]


class ForecastRequest(BaseModel):
    model: ChronosModel = "amazon/chronos-bolt-tiny"
    values: list[float] = Field(min_length=16)
    horizon: int = Field(default=12, ge=1, le=64)


class ForecastResponse(BaseModel):
    available: bool
    model: str
    forecast: list[float] = []
    horizon: int
    reason: str | None = None
