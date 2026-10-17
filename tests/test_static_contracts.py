"""Acceptance suite: synthetic source snippets, exact structural oracles and no execution.

This verifies the static analysis contract, not correctness of every PyTorch model.
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from atlas_parser import parse_paths
from atlas_server import normalize_repo_url


EXAMPLE = '''
class MiniAttention:
    def __init__(self):
        self.q = Linear(8,8)
        self.k = Linear(8,8)
        self.v = Linear(8,8)

    def forward(self, x, mask):
        q = self.q(x).view(1,2,4,8).transpose(1,2)
        k = self.k(x).view(1,2,4,8).transpose(1,2)
        v = self.v(x).view(1,2,4,8).transpose(1,2)
        logits = q @ k.transpose(-2,-1)
        logits = logits.masked_fill(mask, -10000)
        return logits.softmax(-1) @ v

def dynamic_forward(x, enabled):
    if enabled:
        x = x.reshape(1,2,4)
    else:
        x = x.view(1,2,4)
    for step in range(2):
        x = x.transpose(1,2)
    return x

def grpo_style(logp, old_logp, reward):
    adv = (reward - reward.mean()) / (reward.std() + 1e-6)
    ratio = (logp - old_logp).exp()
    return -(ratio * adv).mean()
'''


class StaticAcceptance(unittest.TestCase):
    def make_source(self, text=EXAMPLE):
        d = tempfile.TemporaryDirectory()
        root = Path(d.name)
        path = root / "simulated_model.py"
        path.write_text(text, encoding="utf-8")
        self.addCleanup(d.cleanup)
        return root, path

    def check_graph(self, graph):
        self.assertEqual(graph["schema"], "model-atlas-graph-v1")
        self.assertEqual(graph["meta"]["analysis"], "static-python-ast")
        self.assertFalse(graph["meta"]["runtime_validated"])
        self.assertFalse(graph["meta"]["shape_verified"])
        ids = [n["id"] for n in graph["nodes"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(ids)
        self.assertTrue(all(n.get("source", {}).get("line", 0) > 0 for n in graph["nodes"]))
        self.assertTrue(all(n["output"] == "未由静态 AST 确定" for n in graph["nodes"]))
        for e in graph["edges"]:
            self.assertIn(e["from"], ids)
            self.assertIn(e["to"], ids)
            self.assertIn(e["kind"], ("data", "feedback"))
        for n in graph["nodes"]:
            if n["parent"] is not None:
                self.assertIn(n["parent"], ids)

    def test_attention_every_operation_from_source(self):
        _, source = self.make_source()
        graph = parse_paths([str(source)], entry="MiniAttention.forward")
        self.check_graph(graph)
        labels = [n["label"] for n in graph["nodes"]]
        for expected in ("q : Linear", "k : Linear", "v : Linear",
                         "view()", "transpose()", "MatMult",
                         "masked_fill()", "softmax()", "Return"):
            self.assertTrue(any(expected in x for x in labels), (expected, labels))
        self.assertEqual(sum(n["label"] == "MatMult" for n in graph["nodes"]), 2)
        self.assertGreaterEqual(len([n for n in graph["nodes"] if n["parent"]]), 0)

    def test_two_branches_and_loop_are_not_reported_as_observed(self):
        _, source = self.make_source()
        graph = parse_paths([str(source)], entry="dynamic_forward")
        self.check_graph(graph)
        self.assertEqual(graph["meta"]["conditional_branches"], 1)
        self.assertEqual(graph["meta"]["loops"], 1)
        self.assertTrue(any(n["label"].startswith("φ ") for n in graph["nodes"]))
        self.assertTrue(any(e["kind"] == "feedback" for e in graph["edges"]))
        self.assertTrue(any("Both possible paths" in n["why"] for n in graph["nodes"]))

    def test_training_style_semantics_are_only_heuristics(self):
        _, source = self.make_source()
        graph = parse_paths([str(source)], entry="grpo_style")
        self.check_graph(graph)
        names = " ".join(n["label"] for n in graph["nodes"])
        for fn in ("mean()", "std()", "exp()"):
            self.assertIn(fn, names)
        self.assertFalse(graph["meta"]["runtime_validated"])

    def test_same_input_is_deterministic(self):
        _, source = self.make_source()
        a = parse_paths([str(source)], entry="MiniAttention.forward")
        b = parse_paths([str(source)], entry="MiniAttention.forward")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_source_file_never_runs(self):
        root, source = self.make_source()
        sentinel = root / "executed.txt"
        source.write_text(
            "from pathlib import Path\nPath("+repr(str(sentinel))+").write_text('EXPOSED')\n"
            "def forward(x):\n    return x @ x\n",
            encoding="utf-8")
        graph = parse_paths([str(source)], entry="forward")
        self.check_graph(graph)
        self.assertFalse(sentinel.exists())

    def test_url_validation_rejects_redirect_like_input(self):
        for url in ("http://github.com/a/b", "https://github.com.evil.test/a/b",
                    "https://user@github.com/a/b", "https://github.com:444/a/b",
                    "file:///etc/passwd", "https://evil.test/a/b"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    normalize_repo_url(url)


    def test_dense_source_reports_edge_truncation_instead_of_silent_loss(self):
        # 64 live input parameters × 60 source callsites = 3840 distinct
        # data dependencies; fewer than MAX_NODES, more than MAX_EDGES.
        names = ["x" + str(i) for i in range(64)]
        invocations = "\n".join("    merge(" + ", ".join(names) + ")" for _ in range(60))
        source_text = "def forward(" + ", ".join(names) + "):\n" + invocations + "\n    return x0\n"
        _, source = self.make_source(source_text)
        g = parse_paths([str(source)], entry="forward")
        meta = g["meta"]
        self.assertTrue(meta["truncated"])
        self.assertTrue(meta["truncated_edges"])
        self.assertFalse(meta["truncated_nodes"])
        self.assertGreater(meta["dropped_edges"], 0)
        self.assertEqual(meta["dropped_nodes"], 0)
        self.assertEqual(len(g["edges"]), meta["graph_limits"]["max_edges"])
        self.assertLess(len(g["nodes"]), meta["graph_limits"]["max_nodes"])

    def test_node_overflow_reports_incomplete_graph(self):
        source_text = "def forward(x):\n" + "".join("    touch(x)\n" for _ in range(850)) + "    return x\n"
        _, source = self.make_source(source_text)
        g = parse_paths([str(source)], entry="forward")
        meta = g["meta"]
        self.assertTrue(meta["truncated"])
        self.assertTrue(meta["truncated_nodes"])
        self.assertFalse(meta["truncated_edges"])
        self.assertGreater(meta["dropped_nodes"], 0)
        self.assertEqual(len(g["nodes"]), meta["graph_limits"]["max_nodes"])


if __name__ == "__main__":
    unittest.main()
