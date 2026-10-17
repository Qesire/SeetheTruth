"""Pinned upstream source-vs-graph audit.

Fetches official Hugging Face Transformers 4.44.2 GPT-2 source as *text*;
ast.parse only, never import/exec. Pin is a commit, not a moving main branch.
Enable with SEETHTRUTH_AUDIT_UPSTREAM=1. Normal offline unit tests still run.
"""
from __future__ import annotations
import ast
import hashlib
import os
import tempfile
import unittest
import urllib.request
from pathlib import Path
from atlas_parser import parse_paths

UPSTREAM_COMMIT = "174890280b340b89c5bfa092f6b4fb0e2dc2d7fc"
UPSTREAM_FILE = "src/transformers/models/gpt2/modeling_gpt2.py"
UPSTREAM_URL = f"https://raw.githubusercontent.com/huggingface/transformers/{UPSTREAM_COMMIT}/{UPSTREAM_FILE}"
MAX_SOURCE_BYTES = 250_000


def method(tree, owner, name):
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner)
    return next(n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)


def callsites(method_node, name):
    return sorted(n.lineno for n in ast.walk(method_node) if isinstance(n, ast.Call)
                  and ast.unparse(n.func).split(".")[-1] == name)


def extracted_calls(graph, name):
    return sorted(n["source"]["line"] for n in graph["nodes"] if n.get("callee", "").split(".")[-1] == name)


@unittest.skipUnless(os.environ.get("SEETHTRUTH_AUDIT_UPSTREAM") == "1", "explicit pinned-upstream CI only")
class PinnedOfficialSourceAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        req = urllib.request.Request(UPSTREAM_URL, headers={"User-Agent": "SeetheTruth-Source-Audit/1"})
        with urllib.request.urlopen(req, timeout=25) as reply:
            body = reply.read(MAX_SOURCE_BYTES + 1)
        if len(body) > MAX_SOURCE_BYTES:
            raise AssertionError("upstream file is too large")
        cls.original = body.decode("utf-8")
        cls.tree = ast.parse(cls.original)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name) / "modeling_gpt2.py"
        cls.path.write_text(cls.original, encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def graph(self, owner, name, depth=3):
        result = parse_paths([str(self.path)], entry=owner + "." + name, max_depth=depth)
        self.assertEqual(result["meta"]["source_sha256"]["modeling_gpt2.py"], hashlib.sha256(self.original.encode("utf8")).hexdigest())
        self.assertFalse(result["meta"]["runtime_validated"])
        self.assertFalse(result["meta"]["shape_verified"])
        self.assertFalse(result["meta"]["truncated"],
                         "Truncation means results cannot be presented as complete")
        ids = {n["id"] for n in result["nodes"]}
        self.assertTrue(all(e["from"] in ids and e["to"] in ids for e in result["edges"]))
        return result

    def test_eager_attention_two_distinct_matmuls_and_mask(self):
        original = method(self.tree, "GPT2Attention", "_attn")
        g = self.graph("GPT2Attention", "_attn")
        self.assertEqual(callsites(original, "matmul"), extracted_calls(g, "matmul"))
        self.assertEqual(len(extracted_calls(g, "matmul")), 2)
        for op in ("transpose", "where", "softmax"):
            self.assertEqual(callsites(original, op), extracted_calls(g, op), op)
        # Exactly one symbolic attention-score matmul and one value aggregation
        # for this EAGER method; they are independent source call sites.
        self.assertFalse(any(n.get("certainty") == "runtime_observed" for n in g["nodes"]))

    def test_upcast_baddbmm_is_not_invisible(self):
        original = method(self.tree, "GPT2Attention", "_upcast_and_reordered_attn")
        g = self.graph("GPT2Attention", "_upcast_and_reordered_attn")
        for op in ("baddbmm", "matmul", "reshape", "transpose", "softmax", "where"):
            self.assertEqual(callsites(original, op), extracted_calls(g, op), op)
        self.assertEqual(len(extracted_calls(g, "baddbmm")), 1)
        self.assertEqual(len(extracted_calls(g, "matmul")), 1)
        matched = [n for n in g["nodes"] if n.get("callee") == "torch.baddbmm"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["certainty"], "semantic_rule")
        self.assertIn("bmm", matched[0]["formula"])
        self.assertGreater(g["meta"]["conditional_branches"], 0)

    def test_forward_traverses_both_conditional_self_methods_without_executing(self):
        g = self.graph("GPT2Attention", "forward", depth=2)
        names = g["meta"]["nested_scopes"]
        self.assertIn("GPT2Attention._attn", names)
        self.assertIn("GPT2Attention._upcast_and_reordered_attn", names)
        self.assertEqual(len(extracted_calls(g, "baddbmm")), 1)
        self.assertEqual(len(extracted_calls(g, "matmul")), 3)
        self.assertTrue(any(n.get("condition") for n in g["nodes"]
                            if n.get("callee") in ("self._attn", "self._upcast_and_reordered_attn")))
        self.assertTrue(all(n["certainty"] != "runtime_observed" for n in g["nodes"]))
        # The two routes are mutually conditional alternatives, not 4 GEMMs
        # asserted as one runtime path.
        self.assertTrue(g["meta"]["conditional_branches"] > 0)

    def test_mlp_uses_custom_conv1d_without_asserting_generic_conv_semantics(self):
        g = self.graph("GPT2MLP", "forward")
        projection = [n for n in g["nodes"] if n.get("callee") in ("self.c_fc", "self.c_proj")]
        self.assertEqual(len(projection), 2)
        self.assertTrue(all(n["module_constructor"] == "Conv1D" for n in projection))
        self.assertTrue(all(n["certainty"] == "source_call" for n in projection),
                        "Unknown custom Conv1D must not be falsely interpreted as a PyTorch convolution/GEMM")
        self.assertTrue(all(n["parent"] is None for n in projection))
        self.assertEqual(len(extracted_calls(g, "matmul")), 0)

    def test_sdpa_backend_is_one_source_fused_call(self):
        original = method(self.tree, "GPT2SdpaAttention", "forward")
        g = self.graph("GPT2SdpaAttention", "forward", depth=1)
        self.assertEqual(callsites(original, "scaled_dot_product_attention"),
                         extracted_calls(g, "scaled_dot_product_attention"))
        self.assertEqual(len(extracted_calls(g, "scaled_dot_product_attention")), 1)
        self.assertEqual(len(extracted_calls(g, "matmul")), 0)


if __name__ == "__main__":
    unittest.main()
