# vectormigrate Design Notes

## Product goal

`vectormigrate` exists to make embedding-model migration a repeatable engineering workflow instead of a custom outage project.

## Control plane vs data plane

The release architecture is built around a strict separation:

- Control plane: ABI manifests, migration plans, lifecycle state, evaluation gates
- Data plane: vector namespaces, aliases, embeddings, and query execution

That split matters because backends vary widely, but migration governance should not.

## Why SQLite first

SQLite is the right control-plane default for the current release because it is:

- dependency-light
- easy to inspect locally
- deterministic in tests
- enough for single-user and CI workflows

The design leaves room for a Postgres-backed registry later without changing the domain model.

## Why an in-memory backend first

A real vector backend adapter would dominate the initial codebase and slow iteration. The in-memory backend gives the project a stable harness for:

- alias semantics
- namespace provisioning
- cutover behavior
- retrieval metric validation
- adapter experiments

This keeps the release scope honest. The workflow is already executable before vendor code lands, and new backend adapters can be tested against explicit request-building contracts.

## Why a deterministic embedder

Most migration tools fail to become broadly useful because they require live credentials, expensive re-embedding, or backend-specific setup before a contributor can run anything. The deterministic embedder eliminates that barrier and makes CI meaningful.

## Compatibility module choice

Orthogonal Procrustes was chosen as the first compatibility baseline because it gives the project:

- a research-backed baseline
- no deep learning dependency
- very low inference overhead
- a clear correctness target for tests

It is not the end state. It is the first compatibility primitive around which stronger adapters can be added.

The current codebase now also includes:

- a low-rank affine adapter for a more flexible linear baseline
- a confidence-gated router that decides when to trust adapter search
- a paired-vector dataset format for compatibility experiments

## Hardening additions

The hardening pass added three important control-plane pieces:

- audit events with actor and reason metadata
- rollback support after cutover
- JSON report export for plan review and reproducibility
- structured artifact-bundle export
- stronger input validation and structured CLI errors
- benchmark helpers for latency measurement

These are not optional conveniences. They are part of making migrations inspectable.

## Backend adapter direction

The current backend layer includes testable adapters for OpenSearch, Weaviate, and Qdrant plus pgvector SQL helpers.

They compile migration intents into concrete operations such as:

- index-creation requests
- `_reindex` requests
- alias-swap requests
- k-NN search requests
- named-vector collection creation
- collection alias updates
- pgvector partial-index and query SQL

This keeps backend code explicit and unit-testable even before live cluster integration.

## Community-platform direction

The codebase now includes the first community platform primitives:

- a plugin registry for backend factories
- telemetry hooks for online shadow evaluation
- reproducible JSON artifact bundles
- sample datasets and a walkthrough notebook
- a generated governance template for adapter contribution rules

## MVP invariants

These are the invariants the codebase should keep enforcing:

1. Never mix incompatible embedding ABIs without an explicit routing or adapter decision.
2. Never perform cutover without a persisted plan.
3. Never treat evaluation as optional in the happy path.
4. Keep the control-plane model backend-neutral.
5. Keep documentation, CLI behavior, and tests aligned.

## Expected extension points

- backend adapter protocol
- plan compilers for alias-capable backends
- richer evaluation datasets and judgments
- confidence-aware compatibility routing
- background execution and persistent job state
