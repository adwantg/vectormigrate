from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NoReturn
from uuid import uuid4

from vectormigrate.artifacts import export_run_artifact_bundle
from vectormigrate.benchmarks import benchmark_callable
from vectormigrate.demo import run_demo
from vectormigrate.errors import CLIUsageError, ValidationError, VectorMigrateError
from vectormigrate.models import EmbeddingABI, MigrationPlan, MigrationState
from vectormigrate.plugins import PluginRegistry
from vectormigrate.registry import SQLiteRegistry
from vectormigrate.reporting import export_migration_report
from vectormigrate.validation import validate_alias_name


class StructuredArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise CLIUsageError(message)


def build_parser() -> argparse.ArgumentParser:
    parser = StructuredArgumentParser(
        prog="vectormigrate",
        description="Control-plane tooling for embedding model migration experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo_parser = subparsers.add_parser("demo", help="Run the synthetic MVP demo.")
    demo_parser.add_argument("--db", required=True, help="Path to the SQLite registry file.")

    abi_register = subparsers.add_parser("register-abi", help="Register an embedding ABI.")
    abi_register.add_argument("--db", required=True)
    abi_register.add_argument("--model-id", required=True)
    abi_register.add_argument("--provider", required=True)
    abi_register.add_argument("--version", required=True)
    abi_register.add_argument("--dimensions", required=True, type=int)
    abi_register.add_argument("--distance-metric", default="cosine")
    abi_register.add_argument("--normalization", default="unit")
    abi_register.add_argument("--chunker-version", default="v1")
    abi_register.add_argument("--embedding-scope", default="document")

    abi_list = subparsers.add_parser("list-abis", help="List registered embedding ABIs.")
    abi_list.add_argument("--db", required=True)

    plan_create = subparsers.add_parser("create-plan", help="Create a migration plan.")
    plan_create.add_argument("--db", required=True)
    plan_create.add_argument("--source-abi-id", required=True)
    plan_create.add_argument("--target-abi-id", required=True)
    plan_create.add_argument("--alias-name", required=True)
    plan_create.add_argument("--strategy", default="blue_green")
    plan_create.add_argument("--shadow-percent", default=10.0, type=float)

    plan_list = subparsers.add_parser("list-plans", help="List migration plans.")
    plan_list.add_argument("--db", required=True)

    event_list = subparsers.add_parser("list-events", help="List audit events for a plan.")
    event_list.add_argument("--db", required=True)
    event_list.add_argument("--plan-id", required=True)

    report_export = subparsers.add_parser("export-report", help="Export a migration JSON report.")
    report_export.add_argument("--db", required=True)
    report_export.add_argument("--plan-id", required=True)
    report_export.add_argument("--output", required=True)

    artifact_export = subparsers.add_parser(
        "export-artifacts",
        help="Export a structured artifact bundle.",
    )
    artifact_export.add_argument("--db", required=True)
    artifact_export.add_argument("--plan-id", required=True)
    artifact_export.add_argument("--output-dir", required=True)

    benchmark_parser = subparsers.add_parser(
        "benchmark-demo",
        help="Benchmark the local demo execution path.",
    )
    benchmark_parser.add_argument("--db", required=True)
    benchmark_parser.add_argument("--iterations", type=int, default=5)

    plugin_list = subparsers.add_parser("list-plugins", help="List registered backend plugins.")
    plugin_list.add_argument("--load-entry-points", action="store_true")

    plan_transition = subparsers.add_parser(
        "transition-plan",
        help="Force a plan state transition.",
    )
    plan_transition.add_argument("--db", required=True)
    plan_transition.add_argument("--plan-id", required=True)
    plan_transition.add_argument(
        "--state",
        required=True,
        choices=[state.value for state in MigrationState],
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        registry = SQLiteRegistry(Path(getattr(args, "db", "registry.sqlite")))

        if args.command == "demo":
            payload = run_demo(args.db)
            print(json.dumps(payload, indent=2, sort_keys=True))
            return 0

        if args.command == "register-abi":
            abi = registry.register_abi(
                EmbeddingABI(
                    model_id=args.model_id,
                    provider=args.provider,
                    version=args.version,
                    dimensions=args.dimensions,
                    distance_metric=args.distance_metric,
                    normalization=args.normalization,
                    chunker_version=args.chunker_version,
                    embedding_scope=args.embedding_scope,
                )
            )
            print(json.dumps(abi.to_dict(), indent=2, sort_keys=True))
            return 0

        if args.command == "list-abis":
            print(
                json.dumps(
                    [abi.to_dict() for abi in registry.list_abis()],
                    indent=2,
                    sort_keys=True,
                )
            )
            return 0

        if args.command == "create-plan":
            alias_name = validate_alias_name(args.alias_name)
            plan = registry.create_plan(
                MigrationPlan(
                    source_abi_id=args.source_abi_id,
                    target_abi_id=args.target_abi_id,
                    alias_name=alias_name,
                    strategy=args.strategy,
                    shadow_percent=args.shadow_percent,
                )
            )
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            return 0

        if args.command == "list-plans":
            plans = [plan.to_dict() for plan in registry.list_plans()]
            print(json.dumps(plans, indent=2, sort_keys=True))
            return 0

        if args.command == "list-events":
            events = [event.to_dict() for event in registry.list_events(args.plan_id)]
            print(json.dumps(events, indent=2, sort_keys=True))
            return 0

        if args.command == "export-report":
            path = export_migration_report(registry, args.plan_id, args.output)
            print(json.dumps({"output": str(path)}, indent=2, sort_keys=True))
            return 0

        if args.command == "export-artifacts":
            path = export_run_artifact_bundle(registry, args.plan_id, args.output_dir)
            print(json.dumps({"output": str(path)}, indent=2, sort_keys=True))
            return 0

        if args.command == "benchmark-demo":
            result = benchmark_callable(
                "demo",
                lambda: run_demo(f"{args.db}.{uuid4().hex}.sqlite"),
                args.iterations,
            )
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
            return 0

        if args.command == "list-plugins":
            plugin_registry = PluginRegistry()
            if args.load_entry_points:
                plugin_registry.load_entry_point_plugins()
            plugins_payload = [
                {"name": plugin.name, "description": plugin.description}
                for plugin in plugin_registry.list_plugins()
            ]
            print(json.dumps(plugins_payload, indent=2, sort_keys=True))
            return 0

        if args.command == "transition-plan":
            plan = registry.transition_plan(args.plan_id, MigrationState(args.state))
            print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
            return 0

        parser.error("unknown command")
        return 2
    except (CLIUsageError, ValidationError, VectorMigrateError, KeyError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, sort_keys=True))
        return 2
