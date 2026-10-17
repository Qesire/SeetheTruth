"""Real CPU profiler same-pass accounting and negative evidence oracles."""
import copy
import unittest
try:
    import torch
    from same_pass_evidence import VerifiedPass, bind_same_pass
    from telemetry_capture import TelemetrySession
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is optional outside CPU CI")
class SamePassProfiler(unittest.TestCase):
    def model(self):
        class Branch(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(6, 6)
                self.act = torch.nn.SiLU()
            def forward(self, x):
                return self.act(self.proj(x))
        class Pair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = Branch()
                self.right = Branch()
            def forward(self, x):
                return self.left(x) + self.right(x)
        return Pair()

    def test_nested_exclusive_attribution_and_no_double_counting(self):
        torch.manual_seed(3)
        model = self.model()
        with TelemetrySession(model) as session:
            with torch.no_grad():
                model(torch.ones(4, 6))
                model(torch.ones(4, 6))
        report = session.report()
        calls = report["module_calls"]
        self.assertEqual(len([c for c in calls if c["name"] == "left.proj"]), 2)
        self.assertEqual([c["invocation_index"] for c in calls if c["name"] == "left.proj"], [1, 2])
        self.assertTrue(any(c["exclusive_self_cpu_ms"] > 0 for c in calls))
        self.assertTrue(any(c["exclusive_operator_events"] > 0 for c in calls))
        self.assertTrue(all(c["inclusive_operator_events"] >= c["exclusive_operator_events"]
                            for c in calls))
        roots = [c for c in calls if c["name"] == "<root>"]
        self.assertEqual(len(roots), 2)
        for root in roots:
            children = [c for c in calls if c["parent_invocation_key"] == root["invocation_key"]]
            self.assertEqual({c["name"] for c in children}, {"left", "right"})
            self.assertGreaterEqual(root["inclusive_attributed_cpu_ms"] + 0.001,
                                    sum(c["inclusive_attributed_cpu_ms"] for c in children))
            self.assertGreaterEqual(root["inclusive_operator_events"],
                                    sum(c["inclusive_operator_events"] for c in children))
        exclusive_total = sum(c["exclusive_self_cpu_ms"] for c in calls)
        roots_inclusive = sum(c["inclusive_attributed_cpu_ms"] for c in roots)
        self.assertAlmostEqual(exclusive_total, roots_inclusive, delta=0.01)
        self.assertGreater(sum(c["inclusive_attributed_cpu_ms"] for c in calls),
                           exclusive_total)
        self.assertIsNone(roots[0]["exclusive_self_device_ms"])
        self.assertTrue(report["meta"]["module_inclusive_costs_overlap_and_must_not_be_summed"])

    def test_co_capture_one_pass_and_matched_module_invocations(self):
        model = self.model()
        counter = {"passes": 0}
        def count(_module, _input):
            counter["passes"] += 1
        hook = model.register_forward_pre_hook(count)
        try:
            with VerifiedPass(model, capture_values=False) as capture:
                with torch.no_grad():
                    model(torch.ones(2, 6))
                    model(torch.ones(2, 6))
            report = capture.report()
        finally:
            hook.remove()
        self.assertEqual(counter["passes"], 2)
        self.assertEqual(report["schema"], "seethetruth-correlated-pass-v1")
        self.assertEqual(report["run_id"], report["telemetry"]["meta"]["same_pass_run_id"])
        graph = report["graph"]
        self.assertEqual(report["run_id"], graph["meta"]["same_pass_profile"]["run_id"])
        joined = [n for n in graph["nodes"]
                  if n.get("same_pass_profile", {}).get("status") == "matched"]
        self.assertGreaterEqual(len(joined), 8)
        for n in joined:
            self.assertEqual(n["same_pass_profile"]["run_id"], report["run_id"])
            self.assertIsNotNone(n["same_pass_profile"]["exclusive_self_cpu_ms"])
            self.assertGreaterEqual(n["same_pass_profile"]["inclusive_attributed_cpu_ms"] + .0001,
                                    n["same_pass_profile"]["exclusive_self_cpu_ms"])
        self.assertFalse(report["telemetry"]["meta"]["tensor_values_opt_in"])

    def test_wrong_type_shape_and_truncation_fail_closed(self):
        model = self.model()
        with VerifiedPass(model) as capture:
            with torch.no_grad():
                model(torch.ones(2, 6))
        rt = capture.trace.graph()
        tele = capture.profile.report()
        wrong = copy.deepcopy(tele)
        wrong["module_calls"][0]["type"] = "BadType"
        j = bind_same_pass(rt, wrong, run_id="a", model_name=type(model).__name__)
        self.assertGreater(j["stats"]["type_mismatch"], 0)
        shape = copy.deepcopy(tele)
        shape["module_calls"][0]["last_outputs"] = [{"shape": [999], "dtype": "float32"}]
        j = bind_same_pass(rt, shape, run_id="a", model_name=type(model).__name__)
        self.assertGreater(j["stats"]["tensor_signature_mismatch"], 0)
        truncated = copy.deepcopy(rt)
        truncated["meta"]["truncated"] = True
        j = bind_same_pass(truncated, tele, run_id="a", model_name=type(model).__name__)
        self.assertEqual(j["status"], "blocked_truncated_runtime")
        self.assertFalse(any("same_pass_profile" in n for n in j["graph"]["nodes"]))

    def test_session_is_opt_in_and_single_use(self):
        model = self.model()
        with VerifiedPass(model) as capture:
            self.assertRaises(RuntimeError, capture.report)
        self.assertFalse(capture.trace.graph()["nodes"])
        self.assertEqual(capture.profile.report()["modules"], [])
        with self.assertRaises(RuntimeError):
            with capture:
                pass


if __name__ == "__main__":
    unittest.main()
