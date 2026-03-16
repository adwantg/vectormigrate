<div align="center">
  <h1>💻 vectormigrate Examples</h1>
  <p><em>Concrete, runnable examples for every major feature in the library.</em></p>
</div>

This document gives a concrete example for every major feature in `vectormigrate`.

Runnable grouped examples live in [`examples/`](../examples) and are smoke-tested in the automated suite.

---

## 🏛️ 1. Embedding ABI registry

```python
from vectormigrate import EmbeddingABI, SQLiteRegistry

registry = SQLiteRegistry("/tmp/vectormigrate.sqlite")
abi = registry.register_abi(
    EmbeddingABI(
        model_id="text-embedding-3-large",
        provider="openai",
        version="2026.03",
        dimensions=3072,
        chunker_version="chunks-v1",
    )
)
print(abi.abi_id)
```

## 🗺️ 2. Migration orchestration

```python
from vectormigrate import InMemoryVectorBackend, MigrationOrchestrator, MigrationPlan, SQLiteRegistry

registry = SQLiteRegistry("/tmp/vectormigrate.sqlite")
backend = InMemoryVectorBackend()
orchestrator = MigrationOrchestrator(registry, backend)

plan = orchestrator.create_plan(
    MigrationPlan(
        source_abi_id="openai/text-embedding-3-large@2026.03#chunks-v1",
        target_abi_id="openai/text-embedding-3-large@2026.04#chunks-v1",
        alias_name="retrieval_active",
    )
)
orchestrator.provision_plan(plan.plan_id)
```

## 🧠 3. In-memory backend

```python
import numpy as np

from vectormigrate import Document, EmbeddingABI, InMemoryVectorBackend

abi = EmbeddingABI("demo", "vectormigrate", "v1", dimensions=3)
backend = InMemoryVectorBackend()
backend.create_namespace(abi.abi_id or "", abi)
backend.upsert(
    abi.abi_id or "",
    [Document("doc-1", "hello world")],
    np.array([[1.0, 0.0, 0.0]]),
)
backend.set_alias("active", abi.abi_id or "")
print(backend.search("active", np.array([1.0, 0.0, 0.0]), top_k=1)[0].doc_id)
```

## 🎲 4. Deterministic local embedder

```python
from vectormigrate import DeterministicHashEmbedder, EmbeddingABI

abi = EmbeddingABI("demo", "vectormigrate", "v1", dimensions=8)
embedder = DeterministicHashEmbedder(abi, semantic_salt="shared-space", rotation_seed=7)
vectors = embedder.embed(["embedding migration", "vector aliases"])
print(vectors.shape)
```

## 🧩 5. Compatibility adapters

```python
import numpy as np

from vectormigrate import LowRankAffineAdapter, OrthogonalProcrustesAdapter, ResidualMLPAdapter

source = np.array([[1.0, 0.0], [0.0, 1.0]])
target = np.array([[0.9, 0.1], [0.1, 0.9]])

procrustes = OrthogonalProcrustesAdapter().fit(source, target)
affine = LowRankAffineAdapter(rank=2).fit(source, target)
mlp = ResidualMLPAdapter(hidden_dim=4, epochs=50).fit(source, target)
print(procrustes.transform(source))
print(affine.transform(source))
print(mlp.transform(source))
```

## 📊 6. Paired-vector datasets

```python
from vectormigrate import PairedVectorDataset

dataset = PairedVectorDataset(
    [{"record_id": "pair-1", "source_vector": [1.0, 0.0], "target_vector": [0.9, 0.1]}]
)
dataset.save("/tmp/pairs.jsonl")
loaded = PairedVectorDataset.load("/tmp/pairs.jsonl")
print(loaded.to_matrices()[0].shape)
```

## 🔌 7. Backend adapter compilation

```python
from vectormigrate.backends import OpenSearchAdapter

class FakeTransport:
    def request(self, method, path, body=None):
        return {"method": method, "path": path, "body": body}

adapter = OpenSearchAdapter(FakeTransport())
response = adapter.swap_alias("retrieval_active", "new-index", source_index="old-index")
print(response["path"])
```

## ⚙️ 8. Plugin registry

```python
from vectormigrate import BackendPlugin, PluginRegistry

registry = PluginRegistry()
registry.register(BackendPlugin(name="demo", factory=lambda: "adapter", description="demo backend"))
print(registry.list_plugins()[0].name)
```

## 📡 9. Telemetry and online shadow hooks

```python
from vectormigrate import InMemoryTelemetrySink, OnlineShadowEvaluator

sink = InMemoryTelemetrySink()
evaluator = OnlineShadowEvaluator(sink=sink, top_k=2)
evaluator.record("q1", ["doc-1"], ["doc-1"], {"doc-1": 3.0})
print(evaluator.summary()["query_count"])
```

## 📦 10. Artifact bundles

```python
from vectormigrate import export_run_artifact_bundle

manifest_path = export_run_artifact_bundle(
    registry=registry,
    plan_id=plan.plan_id,
    output_dir="/tmp/vectormigrate-artifacts",
)
print(manifest_path)
```

## 📓 11. Notebook and example assets

```python
from pathlib import Path

from vectormigrate import execute_notebook_smoke, load_documents

docs = load_documents(Path("examples/sample_documents.jsonl"))
globals_dict = execute_notebook_smoke("notebooks/vectormigrate_walkthrough.ipynb", ".")
print(len(docs), len(globals_dict["queries"]))
```

## ⌨️ 12. CLI surfaces

```bash
python -m vectormigrate.cli demo --db /tmp/vectormigrate-demo.sqlite
python -m vectormigrate.cli list-plugins
python -m vectormigrate.cli benchmark-demo --db /tmp/vectormigrate-benchmark --iterations 5
```
