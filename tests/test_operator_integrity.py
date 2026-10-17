"""Independent offline bug regression cases for source-to-operator extraction."""
import tempfile
import unittest
from pathlib import Path

from atlas_parser import parse_paths

SOURCE = '''
class OperatorCases:
    def __init__(self):
        self.proj = Linear(8,8)
    def _scores(self, x, y):
        return x @ y.transpose(-1, -2)
    def forward(self, x, y, use_score):
        z = self.proj(x)
        if use_score:
            z = self._scores(z, y)
        else:
            z = torch.baddbmm(z, z, y)
        z += x
        if use_score:
            newly_defined = z.reshape(2,4)
        result = z if use_score else x
        return result

def branch_only(flag,x):
    if flag:
        z = x.view(2,2)
    return z

def simple_aug(x,y):
    x += y
    return x
'''


class SourceOperatorIntegrity(unittest.TestCase):
    def parse(self, entry):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "op_cases.py"
            p.write_text(SOURCE, encoding="utf8")
            return parse_paths([str(p)], entry=entry, max_depth=3)

    def test_chained_call_in_direct_self_method(self):
        g = self.parse("OperatorCases.forward")
        self.assertIn("OperatorCases._scores", g["meta"]["nested_scopes"])
        scores = next(n for n in g["nodes"] if n.get("callee") == "self._scores")
        childmatmul = [n for n in g["nodes"] if n["parent"] == scores["id"] and n["label"] == "MatMult"]
        self.assertEqual(len(childmatmul), 1)
        self.assertTrue(any(n.get("callee") == "torch.baddbmm" for n in g["nodes"]))
        self.assertTrue(any(n.get("callee") == "self.proj" for n in g["nodes"]))

    def test_cross_function_parameter_binding_edges(self):
        g = self.parse("OperatorCases.forward")
        call = next(n for n in g["nodes"] if n.get("callee") == "self._scores")
        caller_projection = next(n for n in g["nodes"] if n.get("callee") == "self.proj")
        caller_y = next(n for n in g["nodes"] if n["label"] == "y" and n["parent"] is None)
        inner_x = next(n for n in g["nodes"] if n["label"] == "x" and n["parent"] == call["id"])
        inner_y = next(n for n in g["nodes"] if n["label"] == "y" and n["parent"] == call["id"])
        self.assertTrue(any(e["from"] == caller_projection["id"] and e["to"] == inner_x["id"] for e in g["edges"]))
        self.assertTrue(any(e["from"] == caller_y["id"] and e["to"] == inner_y["id"] for e in g["edges"]))
        # The actual nested matrix op must depend on these formal input nodes.
        inner_mul = next(n for n in g["nodes"] if n["label"] == "MatMult" and n["parent"] == call["id"])
        self.assertTrue(any(e["to"] == inner_mul["id"] and e["from"] in {inner_x["id"], inner_y["id"]}
                            for e in g["edges"]))

    def test_augassign_output_connects_to_return(self):
        g = self.parse("simple_aug")
        assign = next(n for n in g["nodes"] if n["label"] == "Add=")
        ret = next(n for n in g["nodes"] if n["label"] == "Return")
        self.assertTrue(any(e["from"] == assign["id"] and e["to"] == ret["id"] for e in g["edges"]))
        input_ids = {n["id"] for n in g["nodes"] if n["label"] in {"x", "y"}}
        self.assertEqual({e["from"] for e in g["edges"] if e["to"] == assign["id"]}, input_ids)

    def test_undefined_branch_must_not_be_definite(self):
        g = self.parse("branch_only")
        candidates = [n for n in g["nodes"] if n["label"] == "φ? z"]
        self.assertEqual(len(candidates), 1)
        self.assertTrue(candidates[0]["conditional_defined"])
        self.assertEqual(candidates[0]["certainty"], "analysis_construct")
        ret = next(n for n in g["nodes"] if n["label"] == "Return")
        self.assertTrue(any(e["from"] == candidates[0]["id"] and e["to"] == ret["id"] for e in g["edges"]))

    def test_inline_conditional_does_not_assert_both_executed(self):
        g = self.parse("OperatorCases.forward")
        conditional = [n for n in g["nodes"] if n["label"] == "φ conditional"]
        self.assertEqual(len(conditional), 1)
        self.assertEqual(conditional[0]["certainty"], "analysis_construct")
        self.assertEqual(conditional[0]["condition"], "use_score")
        self.assertFalse(g["meta"]["runtime_validated"])

    def test_all_edges_reference_real_nodes_and_deterministic(self):
        a = self.parse("OperatorCases.forward")
        b = self.parse("OperatorCases.forward")
        import json
        # absolute temp paths may differ only in source name; all source locators
        # intentionally relative to source directory.
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        ids = {n["id"] for n in a["nodes"]}
        self.assertTrue(all(e["from"] in ids and e["to"] in ids for e in a["edges"]))
        self.assertFalse(a["meta"]["truncated"])


if __name__ == "__main__":
    unittest.main()
