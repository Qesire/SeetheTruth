"""Offline deterministic semantic tests. No torch import, no model execution."""
import tempfile
import unittest
from pathlib import Path
from atlas_parser import parse_paths

SRC = """
class Cond:
    def __init__(self):
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(8, 48))
        self.experts = nn.ModuleList([Expert() for _ in range(4)])
    def forward(self, x, c):
        a, b, g = self.ada(c).chunk(3, dim=-1)
        x = x + g.unsqueeze(1) * (x * a.unsqueeze(1) + b.unsqueeze(1))
        return x

class Route:
    def __init__(self):
        self.router = nn.Linear(8, 4)
        self.experts = nn.ModuleList([Expert() for _ in range(4)])
    def forward(self, x):
        scores, indices = torch.topk(self.router(x), 2, dim=-1)
        output = torch.zeros_like(x)
        for idx in range(4):
            expert_layer = self.experts[idx]
            expert_out = expert_layer(x)
            output.index_add_(0, indices, expert_out)
        return output

def unpack_explicit(a, b):
    left, right = a, b
    return left, right

def produce_pair(x, y):
    query = x + x
    key = y * y
    return query, key

def consume_pair(x, y):
    q, k = produce_pair(x, y)
    return q @ k
"""


class SourceSemantics(unittest.TestCase):
    def build(self, name):
        with tempfile.TemporaryDirectory() as root:
            f = Path(root) / "synthetic.py"
            f.write_text(SRC, encoding="utf-8")
            return parse_paths([str(f)], entry=name, max_depth=2)

    def test_dit_adaln_split_distinct_slots_and_sequential(self):
        g = self.build("Cond.forward")
        slots = [n for n in g["nodes"] if n.get("tuple_binding") in ("a", "b", "g")]
        self.assertEqual(len(slots), 3)
        self.assertEqual({n["tuple_slot"] for n in slots}, {0, 1, 2})
        self.assertEqual(len({n["id"] for n in slots}), 3)
        stages = [n for n in g["nodes"] if "chain_slot" in n]
        self.assertEqual([n["constructor_op"] for n in stages], ["SiLU", "Linear"])
        self.assertEqual([n["certainty"] for n in stages], ["constructor_chain_rule"]*2)
        producer = next(n for n in g["nodes"] if n.get("callee") == "self.ada")
        self.assertTrue(all(n["parent"] == producer["id"] for n in stages))
        self.assertTrue(any(e["from"] == stages[0]["id"] and e["to"] == stages[1]["id"] for e in g["edges"]))
        self.assertTrue(any(e["from"] == stages[1]["id"] and e["to"] == producer["id"] for e in g["edges"]))
        chunk = next(n for n in g["nodes"] if n.get("callee", "").endswith(".chunk"))
        self.assertTrue(all(any(e["from"] == chunk["id"] and e["to"] == n["id"] for e in g["edges"]) for n in slots))

    def test_moe_router_topk_and_mutation(self):
        g = self.build("Route.forward")
        slots = [n for n in g["nodes"] if n.get("tuple_binding") in ("scores", "indices")]
        self.assertEqual({n["tuple_slot"] for n in slots}, {0,1})
        router = next(n for n in g["nodes"] if n.get("callee") == "self.router")
        self.assertEqual(router["certainty"], "constructor_and_semantic_rule")
        self.assertTrue(any("dynamic ModuleList index" in x for x in g["meta"]["unresolved_calls_and_limits"]))
        self.assertFalse(any(n.get("callee") == "self.experts" and n["purpose"] == "Affine/linear projection" for n in g["nodes"]))
        self.assertEqual(len([n for n in g["nodes"] if n.get("callee", "").endswith("index_add_")]),1)
        self.assertEqual(g["meta"]["loops"],1)
        last = next(n for n in g["nodes"] if n["label"] == "Return")
        mutation = next(n for n in g["nodes"] if n.get("callee", "").endswith("index_add_"))
        self.assertTrue(any(e["from"] == mutation["id"] and e["to"] == last["id"] for e in g["edges"]))

    def test_literal_unpack_preserves_per_operand_provenance(self):
        g = self.build("unpack_explicit")
        self.assertFalse(any("unpack[" in n["label"] for n in g["nodes"]))
        a = next(n for n in g["nodes"] if n["label"] == "a")
        b = next(n for n in g["nodes"] if n["label"] == "b")
        ret = next(n for n in g["nodes"] if n["label"] == "Return")
        slots = {n["return_slot"]:n for n in g["nodes"] if "return_slot" in n}
        self.assertEqual(set(slots),{0,1})
        edges={(e["from"],e["to"]) for e in g["edges"]}
        self.assertIn((a["id"],slots[0]["id"]),edges)
        self.assertIn((b["id"],slots[1]["id"]),edges)
        self.assertNotIn((b["id"],slots[0]["id"]),edges)
        self.assertNotIn((a["id"],slots[1]["id"]),edges)
        self.assertIn((slots[0]["id"],ret["id"]),edges)
        self.assertIn((slots[1]["id"],ret["id"]),edges)

    def test_two_distinct_return_slots_bind_to_caller_outputs(self):
        g = self.build("consume_pair")
        call=next(n for n in g["nodes"] if n.get("callee")=="produce_pair")
        returned={n["return_slot"]:n for n in g["nodes"]
                  if n["parent"]==call["id"] and "return_slot" in n}
        self.assertEqual(set(returned),{0,1})
        projections={n["tuple_binding"]:n for n in g["nodes"]
                     if n.get("tuple_binding") in ("q","k") and n["parent"] is None}
        self.assertEqual(set(projections),{"q","k"})
        edge_set={(e["from"],e["to"]) for e in g["edges"]}
        self.assertIn((returned[0]["id"],projections["q"]["id"]),edge_set)
        self.assertIn((returned[1]["id"],projections["k"]["id"]),edge_set)
        self.assertNotIn((returned[1]["id"],projections["q"]["id"]),edge_set)
        self.assertNotIn((returned[0]["id"],projections["k"]["id"]),edge_set)
        mat=next(n for n in g["nodes"] if n["label"]=="MatMult" and n["parent"] is None)
        self.assertIn((projections["q"]["id"],mat["id"]),edge_set)
        self.assertIn((projections["k"]["id"],mat["id"]),edge_set)
        self.assertEqual(g["meta"]["nested_scopes"],["consume_pair","produce_pair"])

    def test_graph_contract_and_repeatability(self):
        a=self.build("Cond.forward")
        b=self.build("Cond.forward")
        import json
        self.assertEqual(json.dumps(a,sort_keys=True),json.dumps(b,sort_keys=True))
        ids={n["id"] for n in a["nodes"]}
        self.assertTrue(all(e["from"] in ids and e["to"] in ids for e in a["edges"]))
        self.assertTrue(all(n.get("source",{}).get("line",0)>0 for n in a["nodes"]))
        self.assertFalse(a["meta"]["runtime_validated"])
        self.assertFalse(a["meta"]["shape_verified"])

if __name__ == "__main__":
    unittest.main()
