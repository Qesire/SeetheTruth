"""Telemetry regression, safety and evidence tests."""
import json
import tempfile
import unittest
from pathlib import Path
try:
    import torch
    from telemetry_capture import TelemetrySession, _sample
except ImportError:
    torch = None

@unittest.skipIf(torch is None, "Requires optional PyTorch")
class TelemetryTests(unittest.TestCase):
    def test_no_implicit_model_execution(self):
        class Counter(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x):
                self.calls += 1
                return x + 1
        m = Counter()
        with TelemetrySession(m) as trace:
            self.assertEqual(m.calls, 0)
        self.assertEqual(m.calls, 0)
        self.assertEqual(trace.report()["modules"], [])
    def test_profiler_and_default_privacy(self):
        torch.manual_seed(5)
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.GELU(), torch.nn.Linear(8, 4))
        x = torch.randn(8, 4)
        with TelemetrySession(model) as profile:
            model(x).square().mean().backward()
        result = profile.report()
        self.assertEqual(result["schema"], "seethetruth-telemetry-v1")
        self.assertFalse(result["meta"]["tensor_values_opt_in"])
        self.assertEqual(result["snapshots"], [])
        self.assertGreater(len(result["operators"]), 0)
        self.assertTrue(any("Linear" in r["type"] for r in result["modules"]))
        self.assertTrue(any(r["last_outputs"] for r in result["modules"]))
        with tempfile.TemporaryDirectory() as d:
            target = profile.save(Path(d) / "profile.json")
            self.assertEqual(json.loads(target.read_text())["schema"], "seethetruth-telemetry-v1")
    def test_sample_bounds_and_nonfinite(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.SiLU())
        with TelemetrySession(model, capture_values=True, max_snapshots=3, max_sample_elements=64) as profile:
            model(torch.randn(12, 4))
        snapshots = profile.report()["snapshots"]
        self.assertTrue(snapshots)
        self.assertLessEqual(len(snapshots), 3)
        for row in snapshots:
            s = row["summary"]
            self.assertLessEqual(s["sampled"], 64)
            self.assertLessEqual(len(s["matrix"]), 20)
            self.assertEqual(len(s["histogram"]["bins"]), 24)
            self.assertEqual(s["finite_count"] + s["nonfinite_count"], s["sampled"])
        x = torch.tensor([[0.0, 1.0, float("nan")], [float("inf"), -2.0, 0.0]])
        v = _sample(x, cap=32)
        self.assertEqual(v["nonfinite_count"], 2)
        self.assertIsNone(v["matrix"][0][2])
        self.assertEqual(sum(v["histogram"]["bins"]), 4)
    def test_limits(self):
        m = torch.nn.Linear(2, 2)
        for kw in ({"max_snapshots":0},{"max_sample_elements":8},{"max_operator_rows":5}):
            with self.assertRaises(ValueError):
                TelemetrySession(m, **kw)

if __name__ == "__main__":
    unittest.main()
