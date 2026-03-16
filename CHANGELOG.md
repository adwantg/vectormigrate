# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2026-03-15

### Added

- stable `EmbeddingABI` contract and public package exports
- SQLite-backed migration control plane with durable plan state
- deterministic in-memory backend and embedder for tests and demos
- evaluation tooling for `Recall@k`, `nDCG@k`, and shadow comparisons
- compatibility adapters: Orthogonal Procrustes, low-rank affine, and residual MLP
- backend request compilers for OpenSearch, Weaviate, Qdrant, and pgvector helpers
- CLI, examples, notebook smoke execution, telemetry hooks, artifacts, plugins, and benchmarks
- architecture, paper-reference, figures, examples, and test-matrix documentation

### Changed

- package metadata now reflects a public `v1.0.0` release
- repository governance now includes a root license, code of conduct, and release guidance

### Notes

- live backend validation and other external-system checks remain documented in `docs/TEST_GAPS.md`
- `v1.0.0` represents the stable public release of the locally verified code surface
