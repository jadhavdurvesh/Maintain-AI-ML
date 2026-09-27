from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from .chronos import MODEL_ID, forecast as chronos_forecast
from .chronos import status as chronos_status
from .schemas import ForecastRequest, ForecastResponse

app = FastAPI(title="MAINTAIN AI ML", version="0.1.3")

UI_FILE = Path(__file__).parent / "static" / "index.html"


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "service": "maintain-ai-ml"}


@app.get("/models")
def models():
    status = chronos_status()
    return {
        "models": {
            "chronos": status,
            "timer": {"available": False, "status": "planned_after_chronos"},
        }
    }


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest):
    if payload.model == "timer":
        raise HTTPException(status_code=501, detail="Timer adapter is not implemented yet")

    status = chronos_status()
    if not status.get("available"):
        return ForecastResponse(
            available=False,
            model=status.get("model", MODEL_ID),
            horizon=payload.horizon,
            reason=status.get("reason", "Chronos runtime is unavailable"),
        )

    try:
        result = chronos_forecast(payload.values, payload.horizon)
        return ForecastResponse(
            available=True,
            model=status.get("model", MODEL_ID),
            forecast=result,
            horizon=payload.horizon,
        )
    except Exception as exc:
        # Return a JSON error instead of letting a slow/failed inference turn
        # into an opaque 502 from the hosting layer.
        return ForecastResponse(
            available=False,
            model=status.get("model", MODEL_ID),
            horizon=payload.horizon,
            reason=f"inference failed: {type(exc).__name__}: {exc}",
        )
