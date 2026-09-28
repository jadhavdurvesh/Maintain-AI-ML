import os


def test_timer_disabled_by_default(monkeypatch):
    monkeypatch.delenv("MAINTAIN_ENABLE_TIMER", raising=False)
    from app import timer

    status = timer.status()
    assert status["available"] is False
    assert status["model"] == "Timer"


def test_timer_requires_enough_history(monkeypatch):
    monkeypatch.setenv("MAINTAIN_ENABLE_TIMER", "1")
    from app import timer

    try:
        timer.forecast([float(i) for i in range(10)], 6)
    except ValueError as exc:
        assert "32" in str(exc)
    else:
        raise AssertionError("Timer should reject insufficient history")
