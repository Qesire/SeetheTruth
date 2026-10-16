"""Static-analysis fixture. The model is deliberately not executed."""
import torch
from torch import nn
import torch.nn.functional as F

class TinyAttention(nn.Module):
    def __init__(self, hidden=64, heads=4):
        super().__init__()
        self.q = nn.Linear(hidden, hidden, bias=False)
        self.k = nn.Linear(hidden, hidden, bias=False)
        self.v = nn.Linear(hidden, hidden, bias=False)
        self.o = nn.Linear(hidden, hidden, bias=False)
        self.heads = heads

    def forward(self, x, mask):
        b, s, h = x.shape
        q = self.q(x).view(b,s,self.heads,-1).transpose(1,2)
        k = self.k(x).view(b,s,self.heads,-1).transpose(1,2)
        v = self.v(x).view(b,s,self.heads,-1).transpose(1,2)
        score = q @ k.transpose(-2,-1)
        score = score.masked_fill(mask, float("-inf"))
        weight = F.softmax(score, dim=-1)
        out = weight @ v
        return self.o(out.transpose(1,2).contiguous().view(b,s,h))

class TinyFFN(nn.Module):
    def __init__(self, hidden=64, ffn=256):
        super().__init__()
        self.gate = nn.Linear(hidden, ffn, bias=False)
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        g = self.gate(x)
        u = self.up(x)
        return self.down(F.silu(g) * u)

class TinyBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = TinyAttention()
        self.ffn = TinyFFN()
        self.norm1 = nn.LayerNorm(64)
        self.norm2 = nn.LayerNorm(64)

    def forward(self, x, mask):
        y = x + self.attn(self.norm1(x), mask)
        return y + self.ffn(self.norm2(y))

def grpo_like_step(policy, batch, optimizer):
    """A learning fixture, not a production GRPO implementation."""
    logits = policy(batch["tokens"])
    logp = F.log_softmax(logits, dim=-1)
    reward = batch["reward"]
    advantage = (reward - reward.mean()) / (reward.std() + 1e-6)
    ratio = (logp - batch["old_logp"]).exp()
    objective = -torch.minimum(ratio * advantage, ratio.clamp(.8,1.2)*advantage).mean()
    optimizer.zero_grad()
    objective.backward()
    optimizer.step()
    return objective
