# Architecture

## Boundary

The main MAINTAIN AI application owns authentication, authorization, machines, telemetry, maintenance, alerts, database access, and safety. This repository owns model loading and inference only.

## Request flow

1. The main backend validates the user and machine scope.
2. It prepares the historical sensor series.
3. It calls `POST /v1/forecast`.
4. This service runs the selected model.
5. The service returns forecast values and model metadata.
6. The main backend exposes the result to Model Lab.

## Why separate it?

Heavy ML dependencies and model weights have different runtime characteristics from a web/API application. A long-running service can load weights once, keep them warm, and scale independently.

## Security

The ML service should not receive Supabase credentials or perform operational writes. Before production use, add service-to-service authentication between the main backend and this service.

## Safety boundary

This service must never issue shutdown commands or decide that a machine is unsafe. Forecasting is advisory. MAINTAIN AI's explicit safety policy and threshold path remain authoritative.
