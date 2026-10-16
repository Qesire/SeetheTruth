import ast
import json
import tempfile
import unittest
from pathlib import Path
from atlas_parser import parse_paths
from atlas_server import normalize_repo_url

ROOT=Path(__file__).resolve().parents[1]
class SourceGraphTests(unittest.TestCase):
    def test_python_compiles(self):
        for path in (ROOT/"atlas_parser.py",ROOT/"atlas_server.py"):
            ast.parse(path.read_text(encoding="utf8"),filename=str(path))
    def test_nested_forward(self):
        g=parse_paths([str(ROOT/"examples"/"tensor_model.py")],entry="TinyBlock.forward")
        self.assertEqual(g["meta"]["entry"],"TinyBlock.forward")
        self.assertFalse(g["meta"]["runtime_validated"])
        self.assertFalse(g["meta"]["shape_verified"])
        self.assertTrue(any(n["parent"] for n in g["nodes"]))
        self.assertTrue(any(n["certainty"]=="constructor_and_semantic_rule" for n in g["nodes"]))
        ids={n["id"] for n in g["nodes"]}
        self.assertTrue(all(e["from"] in ids and e["to"] in ids for e in g["edges"]))
    def test_tensor_operators(self):
        g=parse_paths([str(ROOT/"examples"/"tensor_model.py")],entry="TinyAttention.forward")
        exprs=" ".join(n["label"] for n in g["nodes"])
        for operation in ("view","transpose","masked_fill","softmax","MatMult","contiguous"):
            self.assertIn(operation,exprs)
        for n in g["nodes"]:
            self.assertEqual(n["output"],"未由静态 AST 确定")
    def test_rl(self):
        g=parse_paths([str(ROOT/"examples"/"tensor_model.py")],entry="grpo_like_step")
        names=" ".join(n["label"] for n in g["nodes"])
        for op in ("mean","std","clamp","backward","step"):
            self.assertIn(op,names)
    def test_no_execution(self):
        with tempfile.TemporaryDirectory() as d:
            sentinel=Path(d)/"executed"
            src=Path(d)/"danger.py"
            src.write_text("from pathlib import Path\nPath("+repr(str(sentinel))+").write_text('BAD')\ndef forward(x):\n return x @ x\n",encoding="utf8")
            g=parse_paths([str(src)],entry="forward")
            self.assertFalse(sentinel.exists())
            self.assertTrue(g["nodes"])
    def test_symlink_input_is_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/"repo"
            root.mkdir()
            (root/"good.py").write_text("def forward(x):\n return x + x\n",encoding="utf8")
            outside=Path(d)/"private.py"
            outside.write_text("def secret():\n return 'must not scan'\n",encoding="utf8")
            (root/"link.py").symlink_to(outside)
            graph=parse_paths([str(root)],entry="forward")
            self.assertEqual(graph["meta"]["source_files"],1)
            self.assertTrue(any("symlink skipped" in x for x in graph["meta"]["parse_errors"]))
    def test_restrict_remote_hosts(self):
        self.assertEqual(normalize_repo_url("https://github.com/Qesire/SeetheTruth"),"https://github.com/Qesire/SeetheTruth")
        self.assertEqual(normalize_repo_url("https://huggingface.co/org/model"),"https://huggingface.co/org/model")
        for url in ("http://github.com/a/b","https://evil.example/a/b","https://github.com@evil.example/a/b","file:///etc/passwd"):
            with self.assertRaises(ValueError):normalize_repo_url(url)
if __name__=="__main__":unittest.main()
