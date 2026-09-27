# Model Roadmap

## 1. Chronos-2 — first

Why first:
- It already matches the forecasting capability needed by Model Lab.
- MAINTAIN AI already has a Chronos-2 adapter.
- It is a zero-shot forecaster, so we can validate it before collecting a custom training set.

Done in this phase:
- Stable inference API
- Model caching
- CPU/CUDA selection
- Historical context window
- Forecast horizon
- Health/model status

## 2. Timer — second

Timer should implement the same `/v1/forecast` contract. That lets Model Lab compare models without changing its UI/API contract.

## 3. TimeRadar — third

TimeRadar should be treated as a separate temporal/anomaly capability. Do not force it into the forecast endpoint if its output semantics differ.

## 4. Custom model — later

Only after enough clean telemetry, labels, and a backtesting protocol exist. A custom model should earn its place through measured validation rather than being added just because it is available.

## Evaluation rule

Never turn a forecast into a shutdown decision. Forecasts can support maintenance decisions; the existing explicit safety thresholds remain independent.
