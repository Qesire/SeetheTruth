"""Generate real CPU telemetry for SeetheTruth. Values explicitly sampled."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from telemetry_capture import TelemetrySession
torch.manual_seed(47)
net = torch.nn.Sequential(torch.nn.Linear(16, 40), torch.nn.SiLU(), torch.nn.Linear(40, 16))
optimizer = torch.optim.AdamW(net.parameters(), lr=0.002)
sample = torch.randn(32, 16)
with TelemetrySession(net, capture_values=True, max_snapshots=6, max_sample_elements=512) as profile:
    prediction = net(sample)
    loss = (prediction - sample).square().mean()
    loss.backward()
    optimizer.step()
destination = Path(__file__).with_name("telemetry_demo.json")
profile.save(destination)
print("Created", destination, "operators", len(profile.report()["operators"]), "snapshots", len(profile.report()["snapshots"]))
