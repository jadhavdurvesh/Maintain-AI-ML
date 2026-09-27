from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

BASE_URL = os.environ.get("MAINTAIN_AI_ML_URL", "https://maintain-ai-ml.onrender.com").rstrip("/")

# A deterministic synthetic temperature trend. Chronos only needs a sufficiently
# long history; the goal here is to verify the deployed inference path, not model quality.
VALUES = [
    30.0, 30.2, 30.1, 30.4, 30.3, 30.5, 30.7, 30.6,
    30.8, 31.0, 30.9, 31.1, 31.2, 31.0, 31.3, 31.5,
    31.4, 31.6, 31.7, 31.8, 31.6, 31.9, 32.0, 32.2,
    32.1, 32.4, 32.3, 32.5, 32.7, 32.6, 32.8, 33.0,
]
HORIZON = 12


def request_json(path: str, payload: dict | None = None, timeout: int = 180) -> dict:
    url = f"{BASE_URL}{path}"
    data = None
    headers = {"Accept": "application/json"}
    method = "GET"
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
        method = "POST"

    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
        return json.loads(body)


def main() -> int:
    print(f"Testing deployed MAINTAIN AI ML service: {BASE_URL}")

    # Wait for Render's free service to wake up after a deploy/sleep.
    for attempt in range(1, 16):
        try:
            health = request_json("/health", timeout=60)
            if health.get("status") == "ok":
                print(f"health: OK (attempt {attempt})")
                break
            print(f"health: unexpected response: {health}")
        except Exception as exc:
            print(f"health: waiting ({attempt}/15): {exc}")
        time.sleep(30)
    else:
        print("health check did not become ready", file=sys.stderr)
        return 1

    models = request_json("/models", timeout=60)
    chronos = models.get("models", {}).get("chronos-2", {})
    print(f"models: {chronos}")
    if chronos.get("available") is not True:
        print("Chronos-2 is not available on the deployed service", file=sys.stderr)
        return 1

    payload = {"model": "chronos-2", "values": VALUES, "horizon": HORIZON}

    # Model weights may need to load on the first request. Retry the smoke test
    # rather than treating a cold-start/model-load timeout as a code failure.
    for attempt in range(1, 13):
        try:
            result = request_json("/v1/forecast", payload, timeout=180)
            print("forecast response:")
            print(json.dumps(result, indent=2))

            forecast = result.get("forecast")
            if (
                result.get("available") is True
                and result.get("model") == "chronos-2"
                and result.get("horizon") == HORIZON
                and isinstance(forecast, list)
                and len(forecast) == HORIZON
                and all(isinstance(value, (int, float)) for value in forecast)
            ):
                print("Chronos-2 inference smoke test: PASS")
                return 0

            print(f"inference not ready/valid on attempt {attempt}: {result}")
        except (urllib.error.URLError, TimeoutError, TimeoutError) as exc:
            print(f"inference waiting ({attempt}/12): {exc}")
        except Exception as exc:
            print(f"inference failed ({attempt}/12): {exc}")
        time.sleep(30)

    print("Chronos-2 inference smoke test: FAIL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
