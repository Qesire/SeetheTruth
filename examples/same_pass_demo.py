"""Reproducible CPU same-pass source+runtime+profiler example.

This script explicitly performs its own trusted toy forward once.
No weights, user tensors or values are serialized by default.
"""
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from atlas_parser import parse_paths
from same_pass_evidence import VerifiedPass


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(8, 8)
        self.norm = torch.nn.LayerNorm(8)

    def forward(self, x):
        y = self.proj(x)
        return self.norm(y) + x


def main():
    torch.manual_seed(2026)
    model = Tiny().eval()
    static = parse_paths([str(Path(__file__).resolve())], entry="Tiny.forward", max_depth=1)
    with VerifiedPass(model, source_root=HERE, capture_values=False) as capture:
        with torch.no_grad():
            model(torch.ones(3, 8))
    combined = capture.save(graph_path=HERE / "same_pass_graph.json",
                            telemetry_path=HERE / "same_pass_telemetry.json",
                            static_graph=static)
    print(json.dumps({"run_id": combined["run_id"],
                      "profile": combined["graph"]["meta"]["same_pass_profile"],
                      "source": combined["graph"]["meta"]["source_attribution"]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
