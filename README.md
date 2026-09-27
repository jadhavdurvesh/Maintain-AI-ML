# Maintain AI ML

Dedicated inference service for MAINTAIN AI time-series models.

## Goal
Keep heavy forecasting models out of the main Vercel application while giving Model Lab one stable API for time-series inference.

## Scope
### Phase 1 — Chronos-2
- Primary zero-shot time-series forecaster.
- CPU-first with CUDA when available.
- Cached model loading for a long-running service.
- Forecast-only API; no safety decisions.

### Phase 2 — Timer
- Same forecast contract as Chronos-2.
- Used as a lightweight/reference model and comparison baseline.

### Phase 3 — TimeRadar
TimeRadar is kept separate because its role is temporal/anomaly representation rather than the first forecasting path. Add it only after forecasting is stable.

### Phase 4 — Evaluation / custom models
Backtesting, error metrics, latency, model comparison, and eventually a custom model trained on MAINTAIN AI telemetry when enough clean history exists.

## Explicit non-goals
- No machine CRUD
- No authentication system
- No maintenance scheduling
- No alert creation
- No automatic shutdown decisions
- No direct writes to the operational database
- No frontend

## Architecture
`Browser → MAINTAIN AI backend → Maintain-AI-ML → model → forecast → backend → Model Lab`

The main backend remains the security boundary. It prepares and authorizes the time series; this service performs inference.

## API
- `GET /health`
- `GET /models`
- `POST /v1/forecast`

Example:
```json
{"model":"chronos-2","values":[30.1,30.4,30.2],"horizon":12}
```

## Model order
1. Chronos-2
2. Timer
3. TimeRadar
4. Custom MAINTAIN AI model

## Safety
Forecasts are advisory. Explicit safety thresholds in MAINTAIN AI remain authoritative and must never depend on this service.