"""Pinned Meta DiT -> timm Attention/Mlp cross-module source closure.

Fetches only fixed upstream source bytes and parses ASTs. No model import,
weight loading, execution or guessing of third-party implementation.
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

DIT_REV = "ed81ce2229091fd4ecc9a223645f95cf379d582b"
TIMM_REV = "6e6f3686a7e06bcba37bbd3b7c755f04a516a1e7"
FILES = {
    "models.py": "https://raw.githubusercontent.com/facebookresearch/DiT/" + DIT_REV + "/models.py",
    "timm/models/vision_transformer.py": "https://raw.githubusercontent.com/huggingface/pytorch-image-models/" + TIMM_REV + "/timm/models/vision_transformer.py",
    "timm/layers/__init__.py": "https://raw.githubusercontent.com/huggingface/pytorch-image-models/" + TIMM_REV + "/timm/layers/__init__.py",
    "timm/layers/mlp.py": "https://raw.githubusercontent.com/huggingface/pytorch-image-models/" + TIMM_REV + "/timm/layers/mlp.py",
}


def method(tree, class_name, function_name):
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == function_name)


@unittest.skipUnless(os.environ.get("SEETHTRUTH_AUDIT_UPSTREAM") == "1",
                     "explicit pinned upstream source audit")
class PinnedDitTimmClosure(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.trees = {}
        cls.digests = {}
        for path, url in FILES.items():
            request = urllib.request.Request(url, headers={"User-Agent":"SeetheTruth-pinned-static-audit/1"})
            with urllib.request.urlopen(request, timeout=35) as response:
                raw = response.read(250_001)
            if len(raw) > 250_000:
                raise AssertionError("source size cap exceeded: " + path)
            cls.trees[path] = ast.parse(raw.decode("utf8"), filename=path)
            cls.digests[path] = hashlib.sha256(raw).hexdigest()
            target = cls.root / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
        cls.g = parse_paths([str(cls.root)], entry="DiTBlock.forward", max_depth=1)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_both_external_modules_reach_exact_pinned_files(self):
        g = self.g
        self.assertFalse(g["meta"]["runtime_validated"])
        self.assertFalse(g["meta"]["shape_verified"])
        self.assertFalse(g["meta"]["truncated"])
        self.assertEqual(g["meta"]["source_sha256"], self.digests)
        attn = next(n for n in g["nodes"] if n.get("callee") == "self.attn" and n["parent"] is None)
        mlp = next(n for n in g["nodes"] if n.get("callee") == "self.mlp" and n["parent"] is None)
        self.assertEqual(attn["resolved_source_file"], "timm/models/vision_transformer.py")
        self.assertEqual(mlp["resolved_source_file"], "timm/layers/mlp.py")
        self.assertEqual(attn["resolved_scope"], "Attention.forward")
        self.assertEqual(mlp["resolved_scope"], "Mlp.forward")
        self.assertTrue(any(n["parent"] == attn["id"] for n in g["nodes"]))
        self.assertTrue(any(n["parent"] == mlp["id"] for n in g["nodes"]))

    def test_real_attention_matmul_and_fused_alternative(self):
        path = "timm/models/vision_transformer.py"
        actual_fn = method(self.trees[path], "Attention", "forward")
        expected_infix = sorted(n.lineno for n in ast.walk(actual_fn)
                                if isinstance(n, ast.BinOp) and isinstance(n.op, ast.MatMult))
        expected_sdpa = sorted(n.lineno for n in ast.walk(actual_fn)
                               if isinstance(n, ast.Call)
                               and ast.unparse(n.func).endswith("scaled_dot_product_attention"))
        attn = next(n for n in self.g["nodes"]
                    if n.get("callee") == "self.attn" and n["parent"] is None)
        descendants = [n for n in self.g["nodes"]
                       if n["parent"] == attn["id"] and n["source"]["file"] == path]
        observed_infix = sorted(n["source"]["line"] for n in descendants if n["label"] == "MatMult")
        observed_sdpa = sorted(n["source"]["line"] for n in descendants
                               if n.get("callee","").endswith("scaled_dot_product_attention"))
        self.assertEqual(len(expected_infix), 2)
        self.assertEqual(observed_infix, expected_infix)
        self.assertEqual(observed_sdpa, expected_sdpa)
        self.assertTrue(any(n.get("callee") == "self.qkv" for n in descendants))
        self.assertTrue(any(n.get("callee") == "self.proj" for n in descendants))
        self.assertGreater(self.g["meta"]["conditional_branches"], 0)
        self.assertFalse(self.g["meta"]["runtime_validated"])

    def test_mlp_calls_not_falsely_labeled_as_verified_dense_gemms(self):
        path = "timm/layers/mlp.py"
        actual_fn = method(self.trees[path], "Mlp", "forward")
        expected = sorted(n.lineno for n in ast.walk(actual_fn)
                          if isinstance(n, ast.Call)
                          and ast.unparse(n.func) in ("self.fc1", "self.fc2", "self.act"))
        mlp = next(n for n in self.g["nodes"] if n.get("callee") == "self.mlp" and n["parent"] is None)
        descendants = [n for n in self.g["nodes"] if n["parent"] == mlp["id"]
                       and n["source"]["file"] == path]
        observed = sorted(n["source"]["line"] for n in descendants
                          if n.get("callee") in ("self.fc1", "self.fc2", "self.act"))
        self.assertEqual(observed, expected)
        self.assertFalse(any(n["label"] == "MatMult" for n in descendants))
        self.assertTrue(any(n.get("module_constructor") == "linear_layer" for n in descendants))


if __name__ == "__main__":
    unittest.main()
