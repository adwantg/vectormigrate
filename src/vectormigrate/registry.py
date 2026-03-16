from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from vectormigrate.models import (
    EmbeddingABI,
    MigrationEvent,
    MigrationPlan,
    MigrationState,
    utc_now,
)


class SQLiteRegistry:
    """SQLite-backed control plane for ABIs and migration plans."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS embedding_abis (
                    abi_id TEXT PRIMARY KEY,
                    manifest_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS migration_plans (
                    plan_id TEXT PRIMARY KEY,
                    plan_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS migration_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    event_json TEXT NOT NULL
                )
                """
            )

    def register_abi(self, abi: EmbeddingABI) -> EmbeddingABI:
        self.initialize()
        existing = self.get_abi(abi.abi_id or "")
        if existing is not None and existing.to_dict() != abi.to_dict():
            raise ValueError(f"ABI {abi.abi_id} already exists with different metadata")
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO embedding_abis (abi_id, manifest_json) VALUES (?, ?)",
                (abi.abi_id, json.dumps(abi.to_dict(), sort_keys=True)),
            )
        return abi

    def list_abis(self) -> list[EmbeddingABI]:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT manifest_json FROM embedding_abis ORDER BY abi_id"
            ).fetchall()
        return [EmbeddingABI.from_dict(json.loads(row[0])) for row in rows]

    def get_abi(self, abi_id: str) -> EmbeddingABI | None:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT manifest_json FROM embedding_abis WHERE abi_id = ?",
                (abi_id,),
            ).fetchone()
        if row is None:
            return None
        return EmbeddingABI.from_dict(json.loads(row[0]))

    def create_plan(self, plan: MigrationPlan) -> MigrationPlan:
        self.initialize()
        if self.get_plan(plan.plan_id) is not None:
            raise ValueError(f"plan {plan.plan_id} already exists")
        self.save_plan(plan)
        self.append_event(
            plan_id=plan.plan_id,
            event_type="PLAN_CREATED",
            actor="system",
            to_state=plan.state.value,
            details={"alias_name": plan.alias_name, "strategy": plan.strategy},
        )
        return plan

    def save_plan(self, plan: MigrationPlan) -> MigrationPlan:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR REPLACE INTO migration_plans (plan_id, plan_json) VALUES (?, ?)",
                (plan.plan_id, json.dumps(plan.to_dict(), sort_keys=True)),
            )
        return plan

    def get_plan(self, plan_id: str) -> MigrationPlan | None:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT plan_json FROM migration_plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
        if row is None:
            return None
        return MigrationPlan.from_dict(json.loads(row[0]))

    def list_plans(self) -> list[MigrationPlan]:
        self.initialize()
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT plan_json FROM migration_plans ORDER BY plan_id"
            ).fetchall()
        return [MigrationPlan.from_dict(json.loads(row[0])) for row in rows]

    def transition_plan(
        self,
        plan_id: str,
        new_state: MigrationState,
        actor: str = "system",
        reason: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> MigrationPlan:
        plan = self.require_plan(plan_id)
        previous_state = plan.state.value
        plan.transition(new_state)
        self.save_plan(plan)
        self.append_event(
            plan_id=plan_id,
            event_type="STATE_TRANSITION",
            actor=actor,
            reason=reason,
            from_state=previous_state,
            to_state=new_state.value,
            details=details or {},
        )
        return plan

    def set_plan_state(
        self,
        plan_id: str,
        new_state: MigrationState,
        actor: str = "system",
        reason: str | None = None,
        **updates: Any,
    ) -> MigrationPlan:
        plan = self.require_plan(plan_id)
        previous_state = plan.state.value
        plan.set_state(new_state)
        for key, value in updates.items():
            setattr(plan, key, value)
        plan.updated_at = utc_now()
        self.save_plan(plan)
        self.append_event(
            plan_id=plan_id,
            event_type="STATE_SET",
            actor=actor,
            reason=reason,
            from_state=previous_state,
            to_state=new_state.value,
            details=updates,
        )
        return plan

    def append_event(
        self,
        plan_id: str,
        event_type: str,
        actor: str = "system",
        reason: str | None = None,
        from_state: str | None = None,
        to_state: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> MigrationEvent:
        self.initialize()
        self.require_plan(plan_id)
        event = MigrationEvent(
            plan_id=plan_id,
            event_type=event_type,
            actor=actor,
            reason=reason,
            from_state=from_state,
            to_state=to_state,
            details=details or {},
        )
        with sqlite3.connect(self.path) as connection:
            cursor = connection.execute(
                "INSERT INTO migration_events (plan_id, event_json) VALUES (?, ?)",
                (plan_id, json.dumps(event.to_dict(), sort_keys=True)),
            )
            event_id = cursor.lastrowid
            payload = event.to_dict()
            payload["event_id"] = event_id
            connection.execute(
                "UPDATE migration_events SET event_json = ? WHERE event_id = ?",
                (json.dumps(payload, sort_keys=True), event_id),
            )
        return MigrationEvent.from_dict(payload)

    def list_events(self, plan_id: str) -> list[MigrationEvent]:
        self.initialize()
        self.require_plan(plan_id)
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT event_json FROM migration_events WHERE plan_id = ? ORDER BY event_id",
                (plan_id,),
            ).fetchall()
        return [MigrationEvent.from_dict(json.loads(row[0])) for row in rows]

    def require_plan(self, plan_id: str) -> MigrationPlan:
        plan = self.get_plan(plan_id)
        if plan is None:
            raise KeyError(f"unknown migration plan: {plan_id}")
        return plan

    def require_abi(self, abi_id: str) -> EmbeddingABI:
        abi = self.get_abi(abi_id)
        if abi is None:
            raise KeyError(f"unknown embedding ABI: {abi_id}")
        return abi
