from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-").lower()


class MigrationState(str, Enum):
    DRAFT = "DRAFT"
    PROVISIONED = "PROVISIONED"
    DUAL_WRITE = "DUAL_WRITE"
    BACKFILLING = "BACKFILLING"
    SHADOW_EVAL = "SHADOW_EVAL"
    READY_TO_CUTOVER = "READY_TO_CUTOVER"
    CUTOVER = "CUTOVER"
    HOLDOVER = "HOLDOVER"
    ROLLED_BACK = "ROLLED_BACK"
    DECOMMISSIONED = "DECOMMISSIONED"


ALLOWED_TRANSITIONS: dict[MigrationState, set[MigrationState]] = {
    MigrationState.DRAFT: {MigrationState.PROVISIONED},
    MigrationState.PROVISIONED: {
        MigrationState.DUAL_WRITE,
        MigrationState.BACKFILLING,
        MigrationState.SHADOW_EVAL,
    },
    MigrationState.DUAL_WRITE: {MigrationState.BACKFILLING, MigrationState.SHADOW_EVAL},
    MigrationState.BACKFILLING: {MigrationState.SHADOW_EVAL, MigrationState.DUAL_WRITE},
    MigrationState.SHADOW_EVAL: {MigrationState.BACKFILLING, MigrationState.READY_TO_CUTOVER},
    MigrationState.READY_TO_CUTOVER: {MigrationState.CUTOVER, MigrationState.SHADOW_EVAL},
    MigrationState.CUTOVER: {MigrationState.HOLDOVER, MigrationState.ROLLED_BACK},
    MigrationState.HOLDOVER: {MigrationState.DECOMMISSIONED, MigrationState.ROLLED_BACK},
    MigrationState.ROLLED_BACK: {MigrationState.SHADOW_EVAL, MigrationState.DECOMMISSIONED},
    MigrationState.DECOMMISSIONED: set(),
}


@dataclass(frozen=True)
class EmbeddingABI:
    model_id: str
    provider: str
    version: str
    dimensions: int
    distance_metric: str = "cosine"
    normalization: str = "unit"
    chunker_version: str = "v1"
    tokenizer: str | None = None
    preprocessing_hash: str | None = None
    embedding_scope: str = "document"
    adapter_chain: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)
    metrics: Mapping[str, Any] = field(default_factory=dict)
    abi_id: str | None = None

    def __post_init__(self) -> None:
        if self.dimensions <= 0:
            raise ValueError("dimensions must be positive")
        if self.distance_metric not in {"cosine", "dot", "l2"}:
            raise ValueError("distance_metric must be one of: cosine, dot, l2")
        if self.normalization not in {"none", "l2", "unit"}:
            raise ValueError("normalization must be one of: none, l2, unit")
        if not self.model_id or not self.provider or not self.version:
            raise ValueError("model_id, provider, and version are required")
        if self.abi_id is None:
            object.__setattr__(self, "abi_id", self.generated_abi_id())

    def generated_abi_id(self) -> str:
        provider = slugify(self.provider)
        model = slugify(self.model_id)
        version = slugify(self.version)
        chunker = slugify(self.chunker_version)
        return f"{provider}/{model}@{version}#{chunker}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "version": self.version,
            "dimensions": self.dimensions,
            "distance_metric": self.distance_metric,
            "normalization": self.normalization,
            "chunker_version": self.chunker_version,
            "tokenizer": self.tokenizer,
            "preprocessing_hash": self.preprocessing_hash,
            "embedding_scope": self.embedding_scope,
            "adapter_chain": list(self.adapter_chain),
            "created_at": self.created_at,
            "metrics": dict(self.metrics),
            "abi_id": self.abi_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> EmbeddingABI:
        adapter_chain = tuple(str(item) for item in payload.get("adapter_chain", []))
        return cls(
            model_id=str(payload["model_id"]),
            provider=str(payload["provider"]),
            version=str(payload["version"]),
            dimensions=int(payload["dimensions"]),
            distance_metric=str(payload.get("distance_metric", "cosine")),
            normalization=str(payload.get("normalization", "unit")),
            chunker_version=str(payload.get("chunker_version", "v1")),
            tokenizer=payload.get("tokenizer"),
            preprocessing_hash=payload.get("preprocessing_hash"),
            embedding_scope=str(payload.get("embedding_scope", "document")),
            adapter_chain=adapter_chain,
            created_at=str(payload.get("created_at", utc_now())),
            metrics=payload.get("metrics", {}),
            abi_id=payload.get("abi_id"),
        )


@dataclass
class MigrationPlan:
    source_abi_id: str
    target_abi_id: str
    alias_name: str
    strategy: str = "blue_green"
    shadow_percent: float = 10.0
    plan_id: str = field(default_factory=lambda: f"plan-{uuid4().hex[:12]}")
    state: MigrationState = MigrationState.DRAFT
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    gate_report: Mapping[str, Any] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source_abi_id == self.target_abi_id:
            raise ValueError("source_abi_id and target_abi_id must differ")
        if not 0 <= self.shadow_percent <= 100:
            raise ValueError("shadow_percent must be between 0 and 100")

    def can_transition(self, new_state: MigrationState) -> bool:
        return new_state in ALLOWED_TRANSITIONS[self.state]

    def transition(self, new_state: MigrationState) -> None:
        if not self.can_transition(new_state):
            raise ValueError(f"invalid transition: {self.state} -> {new_state}")
        self.state = new_state
        self.updated_at = utc_now()

    def set_state(self, new_state: MigrationState) -> None:
        self.state = new_state
        self.updated_at = utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_abi_id": self.source_abi_id,
            "target_abi_id": self.target_abi_id,
            "alias_name": self.alias_name,
            "strategy": self.strategy,
            "shadow_percent": self.shadow_percent,
            "plan_id": self.plan_id,
            "state": self.state.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "gate_report": dict(self.gate_report),
            "notes": dict(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MigrationPlan:
        return cls(
            source_abi_id=str(payload["source_abi_id"]),
            target_abi_id=str(payload["target_abi_id"]),
            alias_name=str(payload["alias_name"]),
            strategy=str(payload.get("strategy", "blue_green")),
            shadow_percent=float(payload.get("shadow_percent", 10.0)),
            plan_id=str(payload["plan_id"]),
            state=MigrationState(str(payload.get("state", MigrationState.DRAFT.value))),
            created_at=str(payload.get("created_at", utc_now())),
            updated_at=str(payload.get("updated_at", utc_now())),
            gate_report=payload.get("gate_report", {}),
            notes=payload.get("notes", {}),
        )


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    namespace: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class QueryCase:
    query_id: str
    text: str
    relevance: Mapping[str, float]


@dataclass(frozen=True)
class ShadowMetrics:
    baseline_ndcg_at_k: float
    candidate_ndcg_at_k: float
    delta_ndcg_at_k: float
    baseline_recall_at_k: float
    candidate_recall_at_k: float
    delta_recall_at_k: float
    passes: bool
    top_k: int
    per_query: tuple[Mapping[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_ndcg_at_k": self.baseline_ndcg_at_k,
            "candidate_ndcg_at_k": self.candidate_ndcg_at_k,
            "delta_ndcg_at_k": self.delta_ndcg_at_k,
            "baseline_recall_at_k": self.baseline_recall_at_k,
            "candidate_recall_at_k": self.candidate_recall_at_k,
            "delta_recall_at_k": self.delta_recall_at_k,
            "passes": self.passes,
            "top_k": self.top_k,
            "per_query": list(self.per_query),
        }


@dataclass(frozen=True)
class MigrationEvent:
    plan_id: str
    event_type: str
    actor: str
    reason: str | None = None
    created_at: str = field(default_factory=utc_now)
    from_state: str | None = None
    to_state: str | None = None
    details: Mapping[str, Any] = field(default_factory=dict)
    event_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "event_type": self.event_type,
            "actor": self.actor,
            "reason": self.reason,
            "created_at": self.created_at,
            "from_state": self.from_state,
            "to_state": self.to_state,
            "details": dict(self.details),
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MigrationEvent:
        return cls(
            plan_id=str(payload["plan_id"]),
            event_type=str(payload["event_type"]),
            actor=str(payload.get("actor", "system")),
            reason=payload.get("reason"),
            created_at=str(payload.get("created_at", utc_now())),
            from_state=payload.get("from_state"),
            to_state=payload.get("to_state"),
            details=payload.get("details", {}),
            event_id=int(payload["event_id"]) if payload.get("event_id") is not None else None,
        )
