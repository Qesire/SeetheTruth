# Training as state transitions · SeetheTruth Learning Studio

## Purpose

The Learning Studio is a qualitative mathematics learning product, **not an ML development workbench**, a GPU profiling requirement, or a full autograd implementation. This phase adds **14 new training-specific concept blocks** in a dedicated [training lesson](../training.html). These are not counted among the existing 45 micro-numeric compute blocks: new types have explicit scalar training-state contracts and deep explanations. The two sets are distinct.

## End-to-end training graph

Each optimizer step consists of Parameter Read → `zero_grad(set_to_none=True)` → N microbatch loops, each Data → Forward → Loss → Backward seed → Chain Rule → Correct loss/gradient scaling → Accumulate .grad → Gradient Ready → Optimizer Step. AdamW further expands to first/second moment, bias correction, and **decoupled** weight decay. Two optimizer steps make parameter and optimizer state continuity visible.

Core toy model: θ is a **single trainable scalar**, xᵢ and yᵢ are scalar observations; ŷᵢ=θ·xᵢ and ℓᵢ=½(ŷᵢ−yᵢ)². The half-factor differs from the default PyTorch MSELoss reduction. The gradient is dℓᵢ/dθ=(θxᵢ−yᵢ)·xᵢ, which is independently verified with finite differences. The averaged accumulation result is g=(Σdℓᵢ/dθ)/N **only because all microbatches have one equally weighted valid sample**. This must not be presented as correct for arbitrary variable token counts or masking.

For SGD in this lesson, no momentum or weight decay is used: θ←θ−ηg. AdamW maintains optimizer step t and state m,v, computes m←β₁m+(1−β₁)g and v←β₂v+(1−β₂)g², then bias-corrects by 1−βᵗ. The decoupled weight decay and adaptive step are represented as θ←θ(1−ηλ)−ηm̂/(√v̂+ε). Two steps are checked against an independent hand-derived oracle. A zero gradient with nonzero λ can still change θ under AdamW when a parameter participates in the step.

## Three separate kinds of state

| State | Lifetime | Cleared or updated when |
| --- | --- | --- |
| Model parameter θ | Across all forward passes and steps | Updated once by optimizer after the whole effective batch |
| Grad buffer θ.grad | During one accumulation window | Cleared to None at the beginning; backwards **accumulate** |
| AdamW optimizer m/v and step t | Across optimizer steps | Updated on optimizer.step(), not on each microbatch backward |

`None` and a zero tensor have distinct effects in real PyTorch optimizer semantics for parameters with no gradients. This teaching example always has one participating scalar; it is not an exhaustive simulation of sparse/unreferenced parameter handling.

## Learner experience and evidence

Default mode **Concept** constructs only symbolic event and state descriptions. It explicitly does not compute even toy numeric gradients, moments, or parameters. Selecting **Scalar** permits a bounded one-parameter analytic example. Neither mode requests GPU or external checkpoint. An adversarial counterexample control computes what goes wrong when the loss is not divided by N, or when zero_grad is mistakenly called between backward operations. All 14 stages display formula, precise computation chain, implementation convention, common misconception, falsifiable question, and direct primary-source links.

### Verification

`node tests/training_flow_contracts.js`: validates no numeric output for concept mode, stage alignment, consistent buffered accumulation, theta fixed through microbatches, independent analytic finite-difference gradient, plain SGD update, AdamW's first and second bias-corrected steps, m/v persisted across steps, decoupled decay under g=0 and negative inputs/bad hyperparameters.

`python tests/browser_training_screenshots.py`: launches actual local atlas_server in Chromium, clicks training block inventory, switches symbolic/numeric, checks actual gradient and θ state after AdamW and SGD, tests counterexamples, back/forward and invalid configuration, and captures six user-visible PNG screenshots plus manifest. The GitHub Actions artifact name is `seethetruth-training-gradient-adamw-chromium` (14-day retention). Do not represent these screenshots as PyTorch or actual model training.

### Authoritative references

- [PyTorch Autograd Mechanics](https://docs.pytorch.org/docs/stable/notes/autograd.html): reverse-mode graph and accumulation semantics.
- [PyTorch Optimizer.zero_grad](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html): None versus numerical zero.
- [PyTorch AdamW](https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW.html): moment estimates, bias correction, decoupled decay.
- [PyTorch SGD](https://docs.pytorch.org/docs/stable/generated/torch.optim.SGD.html): basic update rule and opt-in momentum/dampening behavior.
- [PyTorch MSELoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.MSELoss.html): default loss scaling, which differs from our explicitly defined ½ MSE toy.

## Next integration boundary

A future extension may connect tensor-valued symbolic autograd for broadcasting, shared parameters, branching loss, loss masking, variable-length token weighting, activation checkpointing and then distributed synchronization. These are **not** implemented in this delivery. External framework/GPU backends remain optional plugins, not the acceptance standard of the educational app.
