<div align="center">
  <h1>🎓 Toward Safe Embedding Migration</h1>
  <p><em>A formal systems and mathematical model for vector comparability.</em></p>
</div>

This document is a paper-oriented technical draft for `vectormigrate`. It is written as a reference artifact for later manuscript development and is intentionally deeper and more formal than the README.

**Companion references:**
- 📄 [`architecture.md`](architecture.md): first-principles architecture narrative
- 🗺️ [`figures.md`](figures.md): flow and architecture diagrams for paper drafting

---

## 📑 Abstract

Embedding-model migration is a systems problem disguised as a model upgrade. When a retrieval stack changes the embedding model, chunking policy, normalization contract, or index metric, it typically changes the geometry in which nearest-neighbor search is defined. As a result, legacy document vectors and new query vectors are no longer guaranteed to be comparable, and naive migration approaches either incur high operational cost or silently degrade ranking quality. `vectormigrate` addresses this by treating each embedding configuration as a versioned application binary interface, or `EmbeddingABI`, and by separating migration into control-plane, execution-plane, compatibility-plane, and evaluation-plane responsibilities. The design supports blue/green index provisioning, dual-write and backfill, shadow evaluation with explicit gates, atomic cutover where backend aliases exist, and optional compatibility adapters such as Orthogonal Procrustes, low-rank affine maps, and residual multilayer perceptrons. The central systems claim is that safe migration requires explicit compatibility contracts, durable state transitions, and measurable cutover criteria, not ad hoc reindex scripts. This document formalizes the problem, the system model, the method, and the alternative designs considered.

## 🧭 1. Introduction

Modern retrieval systems depend on a latent geometric contract. A document is chunked, preprocessed, embedded into a vector space, indexed under a metric, and later compared against query embeddings that are assumed to live in the same or a justified compatible space. The industry often talks about “switching to a better embedding model” as if the change were equivalent to swapping a pure function. In practice, the change modifies the operational semantics of the retrieval system:

- the vector dimension may change
- the similarity metric may change
- normalization may change
- chunk boundaries may shift
- tokenization and preprocessing may change
- the vector database may expose different query and alias primitives

The result is not just re-computation cost. It is a breach in comparability. If the system compares vectors produced under different contracts without a compatibility mechanism, nearest-neighbor search becomes semantically unstable.

This is the core problem `vectormigrate` solves. The project is not primarily a model SDK. It is a migration system for vector retrieval stacks. Its thesis is that embedding transitions should be managed like a schema migration or ABI transition: versioned, staged, measured, reversible, and backend-aware.

## 🧮 2. Problem Statement

### 📐 2.1 Retrieval from first principles

Let:

- `D = {d_1, ..., d_n}` be a document corpus
- `Q = {q_1, ..., q_m}` be a query distribution
- `c` be a chunking and preprocessing operator
- `f` be an embedding function
- `I` be an approximate nearest-neighbor index
- `s` be a scoring function induced by the metric and normalization policy

A retrieval system returns:

- `R(q) = top_k(s(f(q), f(c(d_i))))`

This compact definition hides a crucial invariant: `f(q)` and `f(c(d_i))` must be comparable under the same scoring regime. That comparability can come from one of four conditions:

1. both vectors are produced by the same embedding contract
2. both spaces are queried independently and fused in a defined way
3. one vector is transformed into the other space by a validated compatibility map
4. both are projected into a canonical space with preserved ranking semantics

If none of these holds, ranking quality is undefined in the strong engineering sense: the system may still return results, but their ordering is not justified by the original retrieval objective.

### 🔄 2.2 Formal migration setting

Suppose the system transitions from an old configuration to a new one.

- `f_old: X -> R^(d_old)`
- `f_new: X -> R^(d_new)`
- `c_old`, `c_new` are chunking operators
- `s_old`, `s_new` are similarity regimes

Legacy serving computes:

- `score_old(q, d) = s_old(f_old(q), f_old(c_old(d)))`

Target serving computes:

- `score_new(q, d) = s_new(f_new(q), f_new(c_new(d)))`

The failure mode during migration is accidental cross-regime scoring:

- `score_mixed(q, d) = s(f_new(q), f_old(c_old(d)))`

or any mixed index that combines `f_old(c_old(d))` and `f_new(c_new(d))` without a contract for score comparability.

### 2.3 Why this is difficult in production

The difficulty is the interaction of three independent sources of complexity:

1. Geometric complexity: embedding spaces drift, rotate, stretch, or change dimensionality.
2. Operational complexity: large corpora require staged backfill and rollback windows.
3. Backend complexity: vector databases expose different capabilities for aliases, multi-vector search, named vectors, reranking, and reindexing.

The engineering problem is therefore to coordinate geometry, operations, and backend capabilities without leaking vendor-specific behavior into the core migration semantics.

## 3. Design Objectives

The system is designed to satisfy six objectives.

### 3.1 Safety

Unsafe mixed-state retrieval must be prevented by construction. The system should make compatibility boundaries explicit and refuse ambiguous transitions.

### 3.2 Auditability

Migration is a long-running state transition, not a single command. The plan, events, gate reports, and cutover rationale must be durable and queryable.

### 3.3 Backend neutrality

The core migration logic should not depend on any single vector database API. Instead, backend-native features should be accessed through capability-driven adapters.

### 3.4 Measurability

Cutovers must be justified through offline and, where possible, online shadow evaluation. Retrieval quality and latency deltas must be observable.

### 3.5 Reversibility

Rollback should be explicit, bounded, and mechanically supported by preserved aliases, legacy indexes, or holdover windows.

### 3.6 Extensibility

The system should allow stronger compatibility modules, new backends, and richer telemetry without destabilizing the core state model.

## 4. System Model

### 4.1 Core entities

`vectormigrate` models migration with the following entities.

#### Embedding ABI

An `EmbeddingABI` is the complete compatibility contract for a vector space:

- model identity
- provider identity
- semantic version
- dimension count
- distance metric
- normalization policy
- chunker version
- tokenizer and preprocessing metadata
- embedding scope
- optional adapter chain

The key idea is that “embedding model version” is too weak to represent compatibility. A change in chunking or normalization changes retrieval semantics even if the model ID stays fixed.

#### Migration Plan

A `MigrationPlan` is a durable transition from `source_abi` to `target_abi`, with explicit state, audit events, and evaluation artifacts.

#### Backend Capability Set

Each backend adapter advertises capabilities such as:

- alias swap support
- named-vector support
- multi-vector query support
- server-side reranking support
- native reindex support

The planner compiles migration intent into backend-native operations using this capability set.

#### Evaluation Bundle

An evaluation bundle contains:

- query sets
- relevance judgments
- shadow run summaries
- gate thresholds
- candidate versus baseline deltas

#### Compatibility Operator

A compatibility operator is an optional transform:

- `T: R^(d_new) -> R^(d_old)` for query-to-legacy bridging
- `T: R^(d_old) -> R^(d_new)` for document backfill approximation
- `T: R^d -> R^k` for a canonical or shared serving space

Examples include Orthogonal Procrustes, low-rank affine maps, and residual MLPs.

### 4.2 Assumptions

The current architecture assumes:

- raw source text or chunks remain available outside the vector index
- operators can provision a target namespace or collection before cutover
- quality can be evaluated on an offline query set, sampled traffic, or both
- the backend adapter can describe which migration primitives are available

The architecture does not assume:

- equal dimensions across models
- backend support for atomic aliasing
- existence of a universal cross-model mapping
- online learning during serving

### 4.3 Failure model

The system explicitly guards against:

- mixed-space scoring without a compatibility contract
- non-audited cutover
- partial backfill interpreted as complete migration
- irreversible switching without holdover
- backend-specific scripts bypassing evaluation gates

The system does not eliminate all risk. It reduces risk by forcing unsafe actions to become explicit, inspectable, and testable.

## 5. Method

### 5.1 Architectural decomposition

The system is decomposed into four planes.

#### Control plane

The control plane persists migration intent and lifecycle state. It stores ABI manifests, migration plans, audit events, and gate reports. In the current codebase, SQLite is used as a lightweight durable substrate for local determinism and CI reproducibility. The deeper design point is not SQLite itself. It is the existence of a durable control state separate from the serving path.

#### Execution plane

The execution plane provisions namespaces, manages dual-write and backfill, compiles backend-native requests, and controls cutover or rollback. This plane owns operational ordering.

#### Compatibility plane

The compatibility plane provides optional transforms that bridge or compare embedding spaces. This plane exists because full re-embedding is sometimes too expensive or too slow for production migration windows.

#### Evaluation plane

The evaluation plane computes metrics such as `nDCG@k` and `Recall@k`, records shadow outcomes, and decides whether the target path can safely replace the baseline.

### 5.2 Why this decomposition is necessary

The decomposition follows from first principles:

- compatibility logic should not be hard-coded into backend adapters
- backend-specific request syntax should not define migration semantics
- evaluation should not be entangled with the serving backend implementation
- control state should not live only in transient scripts or operator memory

This separation is what makes the toolkit portable and research-friendly. Each plane can be reasoned about, benchmarked, and replaced independently.

### 5.3 State machine

The migration plan advances through a strict lifecycle:

`DRAFT -> PROVISIONED -> DUAL_WRITE -> BACKFILLING -> SHADOW_EVAL -> READY_TO_CUTOVER -> CUTOVER -> HOLDOVER or ROLLED_BACK -> DECOMMISSIONED`

This ordering encodes operational invariants:

- no cutover before provisioning
- no decommission before a holdover or explicit confidence
- no “success” without evaluation evidence

### 5.4 Serving strategies

The architecture supports multiple serving strategies because no single method is optimal for all workloads.

#### Strategy A: Blue/green with full re-embedding

Documents are re-embedded into a target namespace. Queries are switched only after the new namespace passes evaluation gates. This is the operationally safest default when raw data is available and re-embedding cost is acceptable.

#### Strategy B: Dual-read with rank fusion

The system queries both old and new spaces during the evaluation phase and fuses candidate sets, for example with reciprocal rank fusion. This allows direct measurement of ranking differences under live-like traffic.

#### Strategy C: Query-time compatibility bridge

The system embeds the query with the new model and applies a transform into the old serving space:

- `q_legacy_hat = T(f_new(q))`

Search is then executed against the old index while a background backfill proceeds. This can dramatically reduce immediate reindex pressure, but it depends on the fidelity of `T`.

#### Strategy D: Confidence-gated hybrid routing

The system estimates whether the compatibility transform is reliable for a given query. If the confidence is high, it uses the transformed query against the old index. Otherwise, it falls back to dual-read or the target index. This creates a quality-preserving bridge between cheap compatibility serving and conservative migration serving.

### 5.5 Compatibility operators

The compatibility operators are intentionally tiered by structural bias.

#### Orthogonal Procrustes

Orthogonal Procrustes solves:

- `min_R ||XR - Y||_F` subject to `R^T R = I`

where `X` and `Y` are paired embeddings. It preserves angles and norms under rotation and reflection. This is a strong baseline when the two spaces differ mostly by rigid transformation.

#### Low-rank affine

Low-rank affine mappings relax the rigid-transform assumption and allow the system to model mild anisotropic drift or dimensional compression with controlled capacity.

#### Residual MLP

A residual MLP provides a nonlinear compatibility map:

- `T(x) = xW_skip + g_theta(x)`

This can fit more complex cross-model relationships, but it introduces a larger risk of overfitting and calibration failure. Its value depends on held-out retrieval quality rather than training loss alone.

### 5.6 Evaluation and cutover rule

Let `M_base` be the baseline metric vector and `M_cand` be the candidate metric vector. A gate function accepts the cutover when:

- `Delta_nDCG@k = nDCG@k_cand - nDCG@k_base >= tau_ndcg`
- `Delta_Recall@k = Recall@k_cand - Recall@k_base >= tau_recall`
- `Delta_latency <= tau_latency`
- no operational safety constraint is violated

In the current toolkit defaults, the thresholds are conservative and configurable. The system is designed so that cutover is a policy decision over measured evidence, not a hidden side effect of backfill completion.

## 6. Why This Solves the Problem

### 6.1 The core insight

The problem is not merely that vectors must be recomputed. The deeper problem is that vector comparison has an implicit contract and existing tooling rarely represents that contract explicitly. `vectormigrate` solves this by making the contract explicit as `EmbeddingABI`, then building every other mechanism around that representation.

Once the contract is explicit:

- mixed-space errors become detectable
- plan semantics become machine-checkable
- backends can be treated as compilers of migration intent
- compatibility modules can be evaluated as swappable hypotheses

### 6.2 The operational insight

Migration safety depends on decoupling cutover from re-embedding. A system that can provision, dual-write, backfill, evaluate, and cut over independently is resilient to partial completion and quality surprises.

### 6.3 The research insight

Compatibility should not be all-or-nothing. Rigid transforms, affine maps, and nonlinear adapters form a ladder of increasingly expressive hypotheses about cross-model alignment. By treating them as pluggable operators behind measurable gates, the architecture supports both practical deployment and publishable empirical comparison.

## 7. Alternative Designs and Why They Were Not Chosen

### 7.1 Hard cutover after full re-embedding only

This approach is simple but brittle. It assumes that a full migration window is operationally affordable and that offline tests are sufficient to catch regressions before traffic sees them. It also offers little flexibility for gradual rollout or compatibility experimentation.

### 7.2 Mixed old and new vectors in one index

This approach appears attractive because it avoids immediate reindex cost, but it destroys score semantics unless a principled comparison rule exists. It is rejected because it creates silent, hard-to-debug ranking corruption.

### 7.3 Universal canonical space as the sole strategy

A canonical space is intellectually appealing, but it assumes stronger invariances than most real systems can justify. It is better treated as an optional research direction than as the only supported operational workflow.

### 7.4 Backend-specific migration scripts

This approach leverages vendor-native features quickly, but it fragments the logic for safety, auditability, and evaluation. Every backend then re-implements the same migration semantics with different bugs and limited comparability.

### 7.5 Converter-only architecture

An offline converter can be powerful, especially for cross-dimensional migration, but it should not replace the migration control plane. Converters are one tool in the compatibility plane, not the entire system.

## 8. Evaluation Protocol for a Publishable Paper

The current toolkit can be the basis for a publishable empirical study if evaluation is structured around the following questions.

### 8.1 Research questions

1. How much retrieval quality is lost when switching embeddings without a compatibility mechanism?
2. Under what conditions can compatibility adapters recover most of the ranking signal?
3. What operational cost reduction is achieved by query-time bridging or converter-assisted backfill?
4. Which backend capabilities materially reduce migration risk or downtime?

### 8.2 Experimental factors

Vary:

- model pair
- dimensionality mismatch
- chunker change severity
- corpus domain shift
- backend capability set
- adapter family
- fallback routing policy

### 8.3 Baselines

Compare against:

- full re-embed blue/green cutover
- dual-read plus fusion without adapter
- Orthogonal Procrustes
- low-rank affine
- residual MLP
- no compatibility bridge

### 8.4 Metrics

Measure:

- `nDCG@k`
- `Recall@k`
- mean reciprocal rank where appropriate
- p50 and p95 latency
- re-embedding cost
- migration wall-clock duration
- rollback frequency or simulated rollback necessity
- calibration quality for confidence-gated routing

### 8.5 Ablations

Recommended ablations include:

- remove `EmbeddingABI` enforcement and measure operator error rates
- disable shadow evaluation gates
- disable confidence gating
- compare backend-native aliasing to client-side routing only
- vary paired-data size for compatibility training

## 9. Limitations

The current implementation still has limits that matter for a paper:

- live backend validation is incomplete without running external clusters
- compatibility quality depends on the availability and representativeness of paired embeddings
- confidence gating is only as good as its calibration data
- offline relevance labels may miss online product behavior

These are not design failures. They are empirical constraints that should be documented explicitly in any manuscript.

## 10. Candidate Publishable Contributions

The project suggests at least three potentially publishable contribution tracks.

### 10.1 Systems contribution

A vendor-neutral control-plane architecture for safe embedding migration, grounded in ABI versioning and measurable cutover rules.

### 10.2 Empirical contribution

A comparative evaluation of compatibility operators and routing policies across model pairs, dimensionality changes, and backend capabilities.

### 10.3 Methodological contribution

A reproducible migration benchmark protocol that evaluates both ranking quality and operational safety, rather than treating migration as a one-dimensional embedding task.

## 11. Practical Examples

### Example A: model upgrade with same dimension but changed chunker

Even if the model dimension remains constant, changing from paragraph chunking to sentence-window chunking changes the distribution of indexed content. `EmbeddingABI` captures that as a compatibility change, so the system forces a real migration plan instead of assuming “same model, same index.”

### Example B: cross-dimensional migration

Suppose a system moves from 1536 dimensions to 3072 dimensions. The backend adapter can provision a new namespace with the correct shape, while the compatibility plane explores whether a low-rank affine map or residual MLP can provide a temporary query bridge during backfill.

### Example C: backend without atomic alias swap

If a backend lacks alias primitives, the architecture still works by elevating cutover to the routing layer. The capability model changes the implementation path, not the migration semantics.

## 12. Summary

`vectormigrate` solves embedding migration by making compatibility explicit, migration state durable, serving strategies modular, and cutover evidence-driven. The design is grounded in a first-principles view of retrieval as a geometric contract and in a systems view of migration as a staged, auditable transition. That combination is what makes the project simultaneously practical for engineering teams and promising as the foundation for a publishable technical paper.
