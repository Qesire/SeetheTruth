# SeetheTruth · Source-grounded guided learning

## Product invariant

A learner should be able to answer **what this operator computes, why it occurs here, and how to examine the result** while remaining in the actual editable model graph. The guide is a navigator and an explanation; its truth comes from the original graph/session and its paper/source knowledge cards.

## Pipeline

1. The visible or free-form graph is validated through the existing LearningLab, GraphSession, GraphLearningPath and operator knowledge cards.
2. `GraphLearningPath.plan(graph,{id})` selects actual source nodes and produces a short question/operation sequence. Curated course hints identify meaningful mechanism boundaries (e.g. Attention Q/attention/output, Wan Self/FFN/writeback). Every hint must resolve against the actual graph; if a graph was edited, route generation uses the current topology.
3. The browser guide selects nodes using `SeetheTruthUnifiedWorkbench.focusNode` and enters numeric mode using the existing selector; execution still uses the normal `#run` action.
4. Numeric completion is accepted only when `SeetheTruthUnifiedWorkbench.inspectAccepted()` exposes a real current-graph `session.trace` and `numericOk`. Selecting a mode, rendering a static card or replaying evidence from another lesson does not complete a numerical observation.
5. The transition into the next course uses the same visible `data-preset` buttons; the next course starts with its own graph and fresh evidence.

## Teaching language

Each stage leads with one relevant **mechanism question**. Follow it with a concrete prediction, the user action, the actual observed input/output or transformation, and the next question. A good first line asks, for example, "Why can the FFN widen channels and return to the original dimension?" It does not begin with an unstructured list of implementation exclusions.

Mathematical guarantees are placed next to the claim to which they apply: source-pinned order and dimensionality are not model-checkpoint parity; numerical path acceptance uses the bounded browser tensor engine; independent PyTorch certificates are scoped to their executed fixture. They remain accessible from the existing cards without dominating onboarding prose.

## Coverage and next phase

The guided graph route applies to Learning Studio preset graphs and freely edited graphs. Focus nodes for FFN, Attention, shared-parameter training, Wan2.2 I2V, image Patch, STFT audio, causal video Conv3D and N-D GQA use source-specific prompts. Other graph lessons use a source-derived topology and operator knowledge prompt. For very small graphs, the guide uses a shorter path.

Specialized Worker, Scheduler, KV state-machine and engineering journeys already have distinct typed evidence. Their guided actions should be added **through the same learning experience core and native state observations**, not treated as ordinary tensor arithmetic. The Android teaching application already shares curriculum and operator sources, but this particular interactive graph guide lives on Learning Studio until an explicit mobile UI route is wired and tested.

## Acceptance

- `node tests/learning_guide_contracts.js`: real graph source nodes, valid proof coordinates, mathematical shape projection, automatic route fallback after graph edits.
- `python tests/browser_guided_onboarding.py`: desktop and phone initial route, numeric mode versus actual execution, lesson progression, curated modality nodes, graph-edit invalidation and JS-console cleanliness.
- The Static Behavior Acceptance workflow stores the rendered screenshot artifact `seethetruth-guided-first-use` for human review; presence of an image alone is not evidence of educational effectiveness.
- Before declaring teaching effectiveness, run a new-user exercise: observe whether learners can predict an output shape, inspect a real value, explain a surprising transformation and transfer the mental model to a second graph. Record observable completion and recovery, not just page clicks.

