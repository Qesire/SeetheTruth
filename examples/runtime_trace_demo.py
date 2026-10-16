"""Explicit opt-in CPU demo. Run this file yourself; the web server never runs it."""
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime_trace import RuntimeTrace


def main():
    torch.manual_seed(7)
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, 4),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randn(3, 8)
    expected = torch.zeros(3, 4)

    with RuntimeTrace(model, max_events=600) as trace:
        prediction = model(tokens)
        loss = (prediction - expected).square().mean()
        trace.backward(loss)
        trace.step(optimizer)

    path = trace.save(Path(__file__).with_name("runtime_trace_demo.json"))
    print("Saved", path, "nodes", len(trace.nodes), "edges", len(trace.edges))
    print("Import the JSON into SeetheTruth's Graph JSON editor.")


if __name__ == "__main__":
    main()
