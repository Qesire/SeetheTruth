"""PyTorch integration tests; skipped on static-only CI without PyTorch."""
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    from runtime_trace import RuntimeTrace
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is an optional runtime tracing dependency")
class RuntimeTraceTests(unittest.TestCase):
    def test_forward_backward_optimizer_and_serialization(self):
        torch.manual_seed(3)
        model = torch.nn.Sequential(
            torch.nn.Linear(4, 6),
            torch.nn.SiLU(),
            torch.nn.Linear(6, 2),
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.02)
        x = torch.randn(3, 4)
        target = torch.zeros(3, 2)
        with RuntimeTrace(model, max_events=500) as trace:
            prediction = model(x)
            loss = (prediction - target).square().mean()
            trace.backward(loss)
            trace.step(optimizer)
        graph = trace.graph()
        self.assertEqual(graph["schema"], "model-atlas-graph-v1")
        self.assertEqual(graph["meta"]["analysis"], "runtime-torch-dispatch")
        self.assertTrue(any(n.get("aten_op", "").startswith("aten.") for n in graph["nodes"]))
        self.assertTrue(any(n.get("phase") == "backward" for n in graph["nodes"]))
        self.assertTrue(any(n.get("updated_count", 0) > 0 for n in graph["nodes"]))
        self.assertTrue(any(n.get("tensor", {}).get("outputs") for n in graph["nodes"]))
        self.assertTrue(all(e["kind"] in {"data", "feedback"} for e in graph["edges"]))
        self.assertEqual(len({n["id"] for n in graph["nodes"]}), len(graph["nodes"]))
        with tempfile.TemporaryDirectory() as tmp:
            saved = trace.save(Path(tmp) / "example-trace.json")
            serialized = json.loads(saved.read_text(encoding="utf8"))
            self.assertEqual(len(serialized["nodes"]), len(graph["nodes"]))
            self.assertTrue(all("values" not in n and "data" not in n for n in serialized["nodes"]))

    def test_opt_in_does_not_call_forward(self):
        class SideEffect(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.called = 0
            def forward(self, x):
                self.called += 1
                return x * 3

        module = SideEffect()
        with RuntimeTrace(module) as trace:
            self.assertEqual(module.called, 0)
        self.assertEqual(module.called, 0)
        self.assertFalse(trace.graph()["nodes"])

    def test_bounds(self):
        with self.assertRaises(ValueError):
            RuntimeTrace(torch.nn.Linear(2, 2), max_events=2)


if __name__ == "__main__":
    unittest.main()
