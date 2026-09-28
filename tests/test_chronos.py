import sys
import types

import torch


def test_forecast_extracts_median_quantile(monkeypatch):
    class FakePipeline:
        quantiles = [0.1, 0.5, 0.9]

        def predict(self, context, prediction_length, **kwargs):
            # [batch, horizon, quantiles]; accept Bolt sampling kwargs too.
            return torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])

    fake_chronos = types.SimpleNamespace(Chronos2Pipeline=object)
    monkeypatch.setitem(sys.modules, "chronos", fake_chronos)

    import app.chronos as chronos
    monkeypatch.setattr(chronos, "get_pipeline", lambda: FakePipeline())

    assert chronos.forecast(list(range(40)), 2) == [2.0, 5.0]
