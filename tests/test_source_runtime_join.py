"""CPU-free adversarial source-to-runtime attribution test oracles."""
from __future__ import annotations
import copy
import unittest
from source_runtime_join import join_graphs


SHA = "a" * 64
BASE = {
    "schema": "model-atlas-graph-v1",
    "name": "SOURCE · Toy.forward",
    "meta": {"analysis": "static-python-ast", "source_sha256": {"toy.py": SHA},
             "truncated": False, "runtime_validated": False},
    "nodes": [
        {"id": "n1", "label": "lin : Linear", "callee": "self.lin",
         "module_constructor": "Linear", "source": {"file": "toy.py", "line": 6}},
        {"id": "n2", "label": "MatMult", "certainty": "source_operator",
         "source": {"file": "toy.py", "line": 7}},
        {"id": "n3", "label": "transpose()", "callee": "x.transpose",
         "source": {"file": "toy.py", "line": 7}},
    ],
    "edges": [],
}
RUNTIME = {
    "schema": "model-atlas-graph-v1",
    "name": "RUNTIME · Toy",
    "meta": {"analysis": "runtime-torch-dispatch", "truncated": False,
             "runtime_validated": True,
             "source_capture": {"enabled": True}},
    "nodes": [
        {"id": "rt1", "phase": "forward", "module": "lin", "module_type": "Linear",
         "invocation_index": 1,
         "source": {"file": "toy.py", "line": 6, "sha256": SHA}},
        {"id": "rt2", "phase": "forward", "aten_op": "aten.mm.default",
         "source": {"file": "toy.py", "line": 7, "sha256": SHA}},
        {"id": "rt3", "phase": "forward", "aten_op": "aten.transpose.int",
         "source": {"file": "toy.py", "line": 7, "sha256": SHA}},
        {"id": "rt4", "phase": "backward", "aten_op": "aten.mm.default",
         "source": {"file": "toy.py", "line": 7, "sha256": SHA}},
    ],
    "edges": [],
}


class SourceRuntimeAttribution(unittest.TestCase):
    def test_exact_module_and_operator_and_nonmutation(self):
        source, runtime = copy.deepcopy(BASE), copy.deepcopy(RUNTIME)
        joined = join_graphs(source, runtime)
        nodes = joined["nodes"]
        self.assertEqual([n.get("source_attribution", {}).get("static_node_id") for n in nodes[:3]],
                         ["n1", "n2", "n3"])
        self.assertNotIn("source_attribution", nodes[3])
        self.assertEqual(joined["meta"]["source_attribution"]["stats"]["matched_module"], 1)
        self.assertEqual(joined["meta"]["source_attribution"]["stats"]["matched_operator"], 2)
        self.assertNotIn("source_attribution", runtime["nodes"][0])
        self.assertEqual(source, BASE)

    def test_wrong_module_type_is_not_matched(self):
        runtime = copy.deepcopy(RUNTIME)
        runtime["nodes"][0]["module_type"] = "CustomLinear"
        joined = join_graphs(BASE, runtime)
        self.assertEqual(joined["nodes"][0]["source_attribution"]["status"], "unmatched")

    def test_wrong_hash_and_line_are_never_matched(self):
        runtime = copy.deepcopy(RUNTIME)
        runtime["nodes"][0]["source"]["sha256"] = "b" * 64
        runtime["nodes"][1]["source"]["line"] = 8
        joined = join_graphs(BASE, runtime)
        self.assertEqual(joined["nodes"][0]["source_attribution"]["status"], "source_mismatch")
        self.assertEqual(joined["nodes"][1]["source_attribution"]["status"], "unmatched")

    def test_same_line_duplicate_matmuls_are_ambiguous(self):
        source = copy.deepcopy(BASE)
        source["nodes"].append({"id": "n4", "label": "MatMult",
                                "certainty": "source_operator",
                                "source": {"file": "toy.py", "line": 7}})
        joined = join_graphs(source, RUNTIME)
        self.assertEqual(joined["nodes"][1]["source_attribution"]["status"], "ambiguous")
        self.assertEqual(joined["nodes"][2]["source_attribution"]["static_node_id"], "n3")

    def test_missing_source_root_causes_explicit_nonattribution(self):
        runtime = copy.deepcopy(RUNTIME)
        runtime["nodes"][0].pop("source")
        joined = join_graphs(BASE, runtime)
        self.assertEqual(joined["nodes"][0]["source_attribution"]["status"], "missing_location")

    def test_truncated_graph_blocks_all_matches(self):
        for side in ("static", "runtime"):
            source = copy.deepcopy(BASE)
            runtime = copy.deepcopy(RUNTIME)
            if side == "static": source["meta"]["truncated"] = True
            else: runtime["meta"]["truncated"] = True
            j = join_graphs(source, runtime)
            self.assertEqual(j["meta"]["source_attribution"]["status"], "blocked_truncated_graph")
            self.assertTrue(all("source_attribution" not in n for n in j["nodes"]))

    def test_operator_identity_rejects_same_line_false_positive(self):
        runtime = copy.deepcopy(RUNTIME)
        runtime["nodes"][1]["aten_op"] = "aten.addmm.default"
        joined = join_graphs(BASE, runtime)
        self.assertEqual(joined["nodes"][1]["source_attribution"]["status"], "unmatched")

    def test_schema_rejection(self):
        with self.assertRaises(ValueError):
            join_graphs({"schema": "wrong"}, RUNTIME)
        with self.assertRaises(ValueError):
            join_graphs(BASE, {"schema": "wrong"})


if __name__ == "__main__":
    unittest.main()
