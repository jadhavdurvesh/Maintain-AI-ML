from typing import Literal
from pydantic import BaseModel, Field

class ForecastRequest(BaseModel):
    model: Literal["chronos-2", "timer"] = "chronos-2"
    values: list[float] = Field(min_length=32)
    horizon: int = Field(default=12, ge=1, le=96)

class ForecastResponse(BaseModel):
    available: bool
    model: str
    forecast: list[float] = []
    horizon: int
    reason: str | None = None
