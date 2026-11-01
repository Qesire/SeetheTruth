# Architecture-first SeetheTruth: curriculum, system modules, and operator bricks

Date: 2026-11-01. Status: initial implementation of an architecture-first **presentation lens**, not a new system execution engine. Applies to the three priority topics: video model sparsity, LLM post-training, and Agent architecture.

## Product contract

SeetheTruth is an **AI Architecture & Infrastructure executable learning workbench**. The first question is not "What does GELU calculate?" but "Which component owns the bottleneck or failure, which architecture choices matter, and how do we prove their effects?"

The default navigation is top-down with a reversible drilldown:

1. **System / SLO** — workload, success/quality, latency, throughput, memory, reliability, cost.
2. **Architecture / topology** — control plane vs data plane, components, responsibilities, queues, storage, ownership.
3. **Runtime / state** — temporal execution, scheduler, worker, KV/cache, optimizer/checkpoint, agent tool events, retries.
4. **Operator / mathematics** — typed tensor graphs, precision, sparsity, gradients, communication and GPU mapping.
5. **Observability / evidence** — per-component events, tensor coordinates, profiler, measured quality, replay and failure cases.

L4 is a *cross-cutting evidence plane*, not an assertion that a metric flows after the kernel. Users can move L4 ↔ any layer and trace why a system-level metric changed.

## Two visible brick types; no semantic conflation

| Display object | Identity and execution | UI | What can be claimed |
| --- | --- | --- | --- |
| **System module** (Scheduler, Worker, Training Pipeline, Agent Workflow) | Existing source-owned specialized lab, with event/state/time semantics; a standalone link/entry until an actual reusable system-node protocol exists | Bounded capsule with subsystem, ownership, resource inputs and observed events | Its specialist source engine and evidence grade, not the presence of a decorative node |
| **Typed operator brick** (LayerNorm, FFN, TopK, KV indexed read, AdamW) | Existing `LearningLab` operator with actual ports, shapes, validation and bounded execution | Draggable graph tile with semantic role, true input port names, source category; canonical operator ID retained | Exactly the existing registered, numeric, autodiff and export capabilities, never simulated GPU performance |

Current implemented operator roles are **Operator/Kernel**, **Representation/Data**, **Runtime/Cache semantics**, and **Training/Optimization**, derived mechanically from source-defined operator categories using `architecture_perspective.js`. Role filters preserve every registered kind, execution contract, free graph, category filter, and code generation behavior. The label `runtime` on a graph brick means **a bounded cache/dispatch operator**, not the entire scheduler.

The shared teaching center's five-layer atlas resolves every launch via existing `teaching_curriculum.js` entries. It neither creates a second curriculum nor registers pseudo-labs. All three paths are available in PC and offline Android.

## Course template (future lessons and rework of the existing 13)

Each system-oriented lesson should begin with:

| Contract | Question |
| --- | --- |
| **Workload and target** | What is the input distribution and measurable SLO or task-quality threshold? |
| **Architecture decision** | Which alternatives compete? Why choose a pipeline, cache policy, route or training objective? |
| **Execution/recovery** | Which state transitions, resource ownership and side effects occur? Can they recover after failure? |
| **Mathematical drilldown** | Which precise kernel, vector operation or gradient supports that behavior? |
| **Counterexample** | Where does the assumption fail (shape, precision, headroom, stale state, group variance, temporal inconsistency)? |
| **Verification** | Is the result analytic, synthetic numeric, pinned source, actual model, or hardware-measured? Which metric judges success? |

### Track A: Video sparsity / inference infrastructure

Start at video SLOs: quality/semantic preservation, temporal dynamics, VRAM and GPU serving cost. Then follow the model pipeline (input conditioning → denoising steps → high/low-noise experts → Blocks/FFN → video decoder), temporal state and inference runtime, channel selection, sparse kernel realizability and measurement. The FFN synthetic channel-contribution oracle is **an offline diagnostic**: it computes the dense activation and W2 contribution before selecting. It does not prove that the same selection can be made cheaply before execution.

Research evaluation requires paired complete videos, prompt/seed manifests, full-block propagation, VBench plus motion/temporal metrics and GPU runtime traces. Local tensor MSE never outranks real video.

### Track B: LLM post-training / training infrastructure

Start at data/rollout workload, throughput, reference model ownership and validation objective; then trace data processing → actor/old/reference policies → reward → loss/gradients → gradient accumulation, optimizer, checkpoint and serving evaluation. Cover SFT/DPO/GRPO and the boundary between data collection, fixed rollout state, parameter updates and off-policy corrections.

Research evaluation needs real model training under pinned revisions, recovery equivalence, held-out tasks, preference win rate and quality/latency/cost. A DPO log-prob derivative is not an optimizer step; a GRPO clipped scalar is not distributed policy training.

### Track C: Agent architecture / orchestration infrastructure

Start at task success, permissions and time budget; then separate planner/supervisor, tool gateway, execution queue, state/checkpoint, worker, verifier, event tracing, failure and approval. Distinguish control graph from Tensor DAG and actual tool side effects from replayable event metadata.

The present Agent demo is a scripted **fault-injection state machine**, not autonomous LLM policy evaluation. Future typed system events should capture run/span/parent IDs, tool schema, idempotency key, checkpoint revision, approvals, resource budget, error codes and verification. Real tool performance requires sandboxed task benchmarks and explicit permissions.

## Interface direction

**Homepage:** top-level three research tracks, each providing System → Design → Runtime → Kernel → Evidence cards. Selecting a layer reveals its architecture question, artifact, evidence limit, and a link to the *existing* exercise. Under it, retain research micro-experiments and the full curriculum library.

**Studio sidebar:** an architecture shelf with real specialist lab routes, followed by typed operator-role filters and the complete existing category filter. Operator tiles expose role, source name and actual input ports. The free graph remains unconstrained by the educational role filter.

**Inspectors (next phase):** for a system module show **ownership, state revisions, causal dependencies, failure/recovery, resource costs and measured traces**. For an operator show **shape, dtype, numerical formula, source revision, VJP and backend eligibility**. Provide meaningful navigation from a source-pinned observed system event to the participating graph node only when that correspondence is actually recorded.

**Visual language:** differentiate data edges, control edges, resource ownership and temporal dependency visually, and distinguish STATIC/SYNTHETIC/PINNED_SOURCE/REAL_CAPTURE/HARDWARE_MEASURED evidence by explicit text. A holographic style may enhance the structural hierarchy but must not conceal contracts, warnings or interactive states.

## External architecture grounding (not direct project implementation evidence)

- [vLLM V1 architecture](https://docs.vllm.ai/en/latest/design/arch_overview/): token-budget scheduler, KV preemption and request-state transitions show why the system layer cannot be reduced to QKV GEMMs.
- [Ray Serve architecture](https://docs.ray.io/en/latest/serve/architecture.html): Controller / Proxy / Replica separation illustrates control plane and data plane and queue ownership.
- [OpenTelemetry GenAI agent spans](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md): workflow/agent/tool spans provide a telemetry reference, with their standard's evolving status respected.
- [LangGraph interrupts](https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/interrupts.mdx): checkpointed interruption and resumed state clarify why durable Agent execution is a system property.

## Acceptance and next steps

**Implemented now:** 3 × 5 architecture questions in the existing shared course home, links to real labs, operator role filter and true port labels in Learning Studio, desktop/offline Android resource sharing, deterministic metadata contract tests and browser acceptance covering track/layer/role transitions.

**Still unimplemented:** a generic editable System IR, cross-system scheduler simulation, typed resource budgeting, time-correlated kernel traces, cross-layer causal jumps backed by real captures, distributed training and real hardware speedups. The next engineering slice should add a **typed, observable system graph** on the existing unified observation spine, using one small existing source engine (Scheduler → KV → Worker) as its first real adapter, and proving replay fidelity before adding other domains.

Do not introduce a new disconnected registry, fabricated traces, performance badges based on FLOP arithmetic, or one-off demos that cannot rejoin the shared microscope.
