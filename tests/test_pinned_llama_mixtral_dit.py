"""Pinned real-code operator concordance across Llama, Mixtral, and DiT.

Downloads TEXT at immutable SHA only; never imports inspected modules, runs
model code, or loads weights. Independent original AST is the call-site oracle.
Set SEETHTRUTH_AUDIT_UPSTREAM=1 to run.
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

TRANSFORMERS_SHA="174890280b340b89c5bfa092f6b4fb0e2dc2d7fc"
DIT_SHA="ed81ce2229091fd4ecc9a223645f95cf379d582b"
URLS={
    "llama":f"https://raw.githubusercontent.com/huggingface/transformers/{TRANSFORMERS_SHA}/src/transformers/models/llama/modeling_llama.py",
    "mixtral":f"https://raw.githubusercontent.com/huggingface/transformers/{TRANSFORMERS_SHA}/src/transformers/models/mixtral/modeling_mixtral.py",
    "dit":f"https://raw.githubusercontent.com/facebookresearch/DiT/{DIT_SHA}/models.py"
}

def func(tree, owner, name):
    candidates=tree.body if owner is None else next(c.body for c in tree.body if isinstance(c,ast.ClassDef) and c.name==owner)
    return next(n for n in candidates if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==name)

def callsite_lines(tree, name):
    return sorted(n.lineno for n in ast.walk(tree) if isinstance(n,ast.Call)
                  and ast.unparse(n.func).split(".")[-1]==name)

def graph_calls(graph,name):
    return sorted(n["source"]["line"] for n in graph["nodes"]
                  if n.get("callee","").split(".")[-1]==name)


@unittest.skipUnless(os.environ.get("SEETHTRUTH_AUDIT_UPSTREAM")=="1","explicit pinned source test")
class RealModelOperatorAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.paths={}
        cls.trees={}
        cls.hashes={}
        for name,url in URLS.items():
            request=urllib.request.Request(url,headers={"User-Agent":"SeetheTruth-AST-Source-Audit/1"})
            with urllib.request.urlopen(request,timeout=25) as response:
                raw=response.read(250_001)
            if len(raw)>250_000:raise AssertionError(name+": source exceeds bounded size")
            body=raw.decode("utf8")
            cls.trees[name]=ast.parse(body)
            cls.hashes[name]=hashlib.sha256(raw).hexdigest()
            path=Path(cls.temp.name)/(name+".py")
            path.write_bytes(raw)
            cls.paths[name]=path

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def graph(self,repo,owner,method,depth=0):
        g=parse_paths([str(self.paths[repo])],entry=(owner+"." if owner else "")+method,max_depth=depth)
        self.assertFalse(g["meta"]["truncated"],f"{repo} {owner}.{method} source graph truncated")
        self.assertEqual(g["meta"]["source_sha256"][repo+".py"],self.hashes[repo])
        self.assertFalse(g["meta"]["runtime_validated"])
        self.assertFalse(g["meta"]["shape_verified"])
        keys={n["id"] for n in g["nodes"]}
        self.assertTrue(all(e["from"] in keys and e["to"] in keys for e in g["edges"]))
        return g

    def concordance(self,repo,owner,method,ops,depth=0):
        source=func(self.trees[repo],owner,method)
        g=self.graph(repo,owner,method,depth)
        for op in ops:
            expected=callsite_lines(source,op)
            actual=graph_calls(g,op)
            self.assertEqual(actual,expected,f"{repo}.{owner}.{method}:{op} missing or doubled")
        return g

    def test_llama_gqa_paths_and_rope_are_source_accurate(self):
        g=self.concordance("llama","LlamaAttention","forward",[
            "linear","matmul","repeat_kv","apply_rotary_pos_emb","view",
            "transpose","softmax","cat","reshape",
            "q_proj","k_proj","v_proj","o_proj"
        ])
        self.assertEqual(len(graph_calls(g,"matmul")),2)
        self.assertEqual(len(graph_calls(g,"repeat_kv")),2)
        self.assertEqual(len(graph_calls(g,"apply_rotary_pos_emb")),1)
        for name in ("q_proj","k_proj","v_proj","o_proj"):
            n=next(n for n in g["nodes"] if n.get("callee")=="self."+name)
            self.assertEqual(n["module_constructor"],"Linear")
            self.assertEqual(n["certainty"],"constructor_and_semantic_rule")
        self.assertGreater(g["meta"]["conditional_branches"],0)

    def test_llama_rope_and_head_expansion_helpers(self):
        rope=self.concordance("llama",None,"apply_rotary_pos_emb",["rotate_half","unsqueeze"])
        self.assertEqual(len(graph_calls(rope,"rotate_half")),2)
        repeat=self.concordance("llama",None,"repeat_kv",["expand","reshape"])
        self.assertEqual(len(graph_calls(repeat,"expand")),1)
        self.assertEqual(len(graph_calls(repeat,"reshape")),1)
        self.assertEqual(repeat["meta"]["conditional_branches"],1)

    def test_mixtral_router_routing_mutation_and_expert_ffn(self):
        moe=self.concordance("mixtral","MixtralSparseMoeBlock","forward",[
            "gate","topk","softmax","one_hot","index_add_","reshape","view"
        ])
        self.assertEqual(len(graph_calls(moe,"topk")),1)
        self.assertEqual(len(graph_calls(moe,"index_add_")),1)
        slot_nodes=[n for n in moe["nodes"] if n.get("tuple_binding") in ("routing_weights","selected_experts")]
        self.assertEqual({n["tuple_slot"] for n in slot_nodes},{0,1})
        self.assertEqual(len({n["id"] for n in slot_nodes}),2)
        self.assertTrue(any("dynamic ModuleList index" in u for u in moe["meta"]["unresolved_calls_and_limits"]))
        expert=self.concordance("mixtral","MixtralBlockSparseTop2MLP","forward",["w1","w2","w3"])
        self.assertEqual(len([n for n in expert["nodes"] if n.get("module_constructor")=="Linear"]),3)
        self.assertEqual(len(graph_calls(expert,"w2")),1)
        self.assertFalse(any(n.get("certainty")=="runtime_observed" for n in expert["nodes"]))

    def test_dit_adaln_sequential_and_gated_residual(self):
        dit=self.concordance("dit","DiTBlock","forward",[
            "adaLN_modulation","chunk","norm1","norm2","attn","mlp","modulate","unsqueeze"
        ])
        names=("shift_msa","scale_msa","gate_msa","shift_mlp","scale_mlp","gate_mlp")
        slots=[n for n in dit["nodes"] if n.get("tuple_binding") in names]
        self.assertEqual(len(slots),6)
        self.assertEqual({n["tuple_slot"] for n in slots},set(range(6)))
        self.assertEqual({n["tuple_binding"] for n in slots},set(names))
        stages=[n for n in dit["nodes"] if "chain_slot" in n]
        self.assertEqual([n["constructor_op"] for n in stages],["SiLU","Linear"])
        self.assertEqual(stages[1]["certainty"],"constructor_chain_rule")
        self.assertEqual(len(graph_calls(dit,"modulate")),2)
        self.assertEqual(len(graph_calls(dit,"attn")),1)
        self.assertEqual(len(graph_calls(dit,"mlp")),1)
        self.assertEqual(len(graph_calls(dit,"matmul")),0)
        for custom in ("attn","mlp"):
            n=next(n for n in dit["nodes"] if n.get("callee")=="self."+custom)
            self.assertEqual(n["certainty"],"source_call")
        helper=self.concordance("dit",None,"modulate",["unsqueeze"])
        self.assertEqual(len(graph_calls(helper,"unsqueeze")),2)

    def test_dit_final_modulation_two_way_split(self):
        g=self.concordance("dit","FinalLayer","forward",["adaLN_modulation","chunk","linear","modulate"])
        chunks=[n for n in g["nodes"] if n.get("tuple_binding") in ("shift","scale")]
        self.assertEqual({n["tuple_slot"] for n in chunks},{0,1})
        self.assertEqual(len([n for n in g["nodes"] if n.get("constructor_op")=="Linear"]),1)

if __name__=="__main__":
    unittest.main()
