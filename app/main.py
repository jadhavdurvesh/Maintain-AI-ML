from pathlib import Path
from threading import Lock

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .chronos import MODEL_ID as CHRONOS_MODEL_ID, forecast as chronos_forecast
from .chronos import status as chronos_status
from .schemas import ForecastRequest, ForecastResponse
from .timer import forecast as timer_forecast
from .timer import status as timer_status

app = FastAPI(title="MAINTAIN AI ML", version="0.3.0")

# The ML service is deployed separately from the main Vercel application.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://([a-zA-Z0-9-]+\.)?vercel\.app$",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

UI_FILE = Path(__file__).parent / "static" / "index.html"
# Free Render instances are memory constrained. Only one model may perform
# inference at a time, and each model releases itself when it is done.
_INFERENCE_LOCK = Lock()


@app.get("/", include_in_schema=False)
def dashboard():
    return FileResponse(UI_FILE)


@app.get("/health")
def health():
    return {"status": "ok", "service": "maintain-ai-ml"}


@app.get("/models")
def models():
    return {"models": {"chronos": chronos_status(), "timer": timer_status()}}


@app.post("/v1/forecast", response_model=ForecastResponse)
def forecast(payload: ForecastRequest):
    if payload.horizon < 1 or payload.horizon > 64:
        return ForecastResponse(available=False, model=str(payload.model), horizon=payload.horizon, reason="horizon must be between 1 and 64")

    with _INFERENCE_LOCK:
        if payload.model == "timer":
            status = timer_status()
            if not status.get("available"):
                return ForecastResponse(available=False, model="Timer", horizon=payload.horizon, reason=status.get("reason", "Timer is unavailable"))
            try:
                result = timer_forecast(payload.values, payload.horizon)
                return ForecastResponse(available=True, model="Timer", forecast=result, horizon=payload.horizon)
            except Exception as exc:
                return ForecastResponse(available=False, model="Timer", horizon=payload.horizon, reason=f"inference failed: {type(exc).__name__}: {exc}")

        status = chronos_status()
        if not status.get("available"):
            return ForecastResponse(available=False, model=status.get("model", CHRONOS_MODEL_ID), horizon=payload.horizon, reason=status.get("reason", "Chronos runtime is unavailable"))
        try:
            result = chronos_forecast(payload.values, payload.horizon)
            return ForecastResponse(available=True, model=status.get("model", CHRONOS_MODEL_ID), forecast=result, horizon=payload.horizon)
        except Exception as exc:
            return ForecastResponse(available=False, model=status.get("model", CHRONOS_MODEL_ID), horizon=payload.horizon, reason=f"inference failed: {type(exc).__name__}: {exc}")
