"""Non-executing multi-file import resolution regressions.

An explicitly imported class must bind to its exact file even when other
source files declare the same name. Missing sources remain unresolved.
"""
import tempfile
import unittest
from pathlib import Path
from atlas_parser import parse_paths


class SourceImports(unittest.TestCase):
    def fixture(self, import_line="from pkg.layers import Attention as ImportedAttention",
                constructor="ImportedAttention"):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        pkg = root / "pkg"
        pkg.mkdir()
        (pkg / "layers.py").write_text(
            "class Attention:\n"
            "    def forward(self, x):\n"
            "        return x @ x\n", encoding="utf8")
        (pkg / "decoy.py").write_text(
            "class Attention:\n"
            "    def forward(self, x):\n"
            "        return x + 1\n", encoding="utf8")
        (pkg / "entry.py").write_text(
            import_line + "\n"
            "class Block:\n"
            "    def __init__(self):\n"
            "        self.attn = " + constructor + "()\n"
            "    def forward(self, x):\n"
            "        return self.attn(x)\n", encoding="utf8")
        return root

    def test_exact_import_alias_wins_over_decoy_class(self):
        root = self.fixture()
        g = parse_paths([str(root)], entry="Block.forward", max_depth=2)
        inv = next(n for n in g["nodes"] if n.get("callee") == "self.attn")
        self.assertEqual(inv["resolved_source_file"], "pkg/layers.py")
        self.assertEqual(inv["resolved_scope"], "pkg.layers.Attention.forward")
        actual = [n for n in g["nodes"] if n["parent"] == inv["id"]]
        self.assertTrue(actual)
        self.assertTrue(all(n["source"]["file"] == "pkg/layers.py" for n in actual))
        self.assertTrue(any(n["label"] == "MatMult" for n in actual))
        self.assertFalse(any(n["label"] == "Add" for n in actual))
        self.assertFalse(g["meta"]["runtime_validated"])
        self.assertFalse(g["meta"]["shape_verified"])

    def test_unimported_external_class_not_guessed(self):
        root = self.fixture(import_line="# deliberately no import", constructor="Attention")
        g = parse_paths([str(root)], entry="Block.forward", max_depth=3)
        inv = next(n for n in g["nodes"] if n.get("callee") == "self.attn")
        self.assertNotIn("resolved_scope", inv)
        self.assertFalse(any(n["parent"] == inv["id"] for n in g["nodes"]))

    def test_imported_but_missing_source_is_explicitly_unresolved(self):
        root = self.fixture()
        g = parse_paths([str(root / "pkg" / "entry.py")], entry="Block.forward", max_depth=2)
        self.assertTrue(any("unloaded imported module: pkg.layers.Attention" in x
                            for x in g["meta"]["unresolved_calls_and_limits"]))
        inv = next(n for n in g["nodes"] if n.get("callee") == "self.attn")
        self.assertNotIn("resolved_source_file", inv)

    def test_target_file_hash_and_parameter_binding(self):
        root = self.fixture()
        g = parse_paths([str(root)], entry="Block.forward", max_depth=2)
        inv = next(n for n in g["nodes"] if n.get("callee") == "self.attn")
        self.assertIn("pkg/layers.py", g["meta"]["source_sha256"])
        param = next(n for n in g["nodes"]
                     if n["parent"] == inv["id"] and n["label"] == "x")
        self.assertTrue(any(e["to"] == param["id"] for e in g["edges"]))


if __name__ == "__main__":
    unittest.main()
