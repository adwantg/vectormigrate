from __future__ import annotations

import json
from pathlib import Path

from vectormigrate.cli import main


def test_cli_registers_abi_and_lists_it(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "registry.sqlite"

    assert (
        main(
            [
                "register-abi",
                "--db",
                str(db_path),
                "--model-id",
                "legacy",
                "--provider",
                "vectormigrate",
                "--version",
                "2026.03",
                "--dimensions",
                "16",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert main(["list-abis", "--db", str(db_path)]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["model_id"] == "legacy"


def test_cli_demo_runs(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "demo.sqlite"

    assert main(["demo", "--db", str(db_path)]) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["plan_state"] == "CUTOVER"
    assert payload["shadow_metrics"]["passes"] is True


def test_cli_lists_events_and_exports_report(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "registry.sqlite"

    assert (
        main(
            [
                "register-abi",
                "--db",
                str(db_path),
                "--model-id",
                "legacy",
                "--provider",
                "vectormigrate",
                "--version",
                "2026.03",
                "--dimensions",
                "16",
            ]
        )
        == 0
    )
    first_abi = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "register-abi",
                "--db",
                str(db_path),
                "--model-id",
                "upgrade",
                "--provider",
                "vectormigrate",
                "--version",
                "2026.04",
                "--dimensions",
                "16",
            ]
        )
        == 0
    )
    second_abi = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "create-plan",
                "--db",
                str(db_path),
                "--source-abi-id",
                first_abi["abi_id"],
                "--target-abi-id",
                second_abi["abi_id"],
                "--alias-name",
                "active",
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)

    assert main(["list-events", "--db", str(db_path), "--plan-id", plan["plan_id"]]) == 0
    events = json.loads(capsys.readouterr().out)
    assert events[0]["event_type"] == "PLAN_CREATED"

    output_path = tmp_path / "report.json"
    assert (
        main(
            [
                "export-report",
                "--db",
                str(db_path),
                "--plan-id",
                plan["plan_id"],
                "--output",
                str(output_path),
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"] == str(output_path)
    exported = json.loads(output_path.read_text(encoding="utf-8"))
    assert exported["plan"]["plan_id"] == plan["plan_id"]

    artifact_dir = tmp_path / "artifacts"
    assert (
        main(
            [
                "export-artifacts",
                "--db",
                str(db_path),
                "--plan-id",
                plan["plan_id"],
                "--output-dir",
                str(artifact_dir),
            ]
        )
        == 0
    )
    artifact_payload = json.loads(capsys.readouterr().out)
    manifest_path = Path(artifact_payload["output"])
    assert manifest_path.name == "artifact_manifest.json"


def test_cli_validates_alias_and_supports_artifacts_and_benchmark(tmp_path: Path, capsys) -> None:
    db_path = tmp_path / "registry.sqlite"

    exit_code = main(
        [
            "create-plan",
            "--db",
            str(db_path),
            "--source-abi-id",
            "old",
            "--target-abi-id",
            "new",
            "--alias-name",
            " bad alias ",
        ]
    )
    error_payload = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert "alias_name" in error_payload["error"]

    assert main(["benchmark-demo", "--db", str(tmp_path / "demo.sqlite"), "--iterations", "2"]) == 0
    benchmark_payload = json.loads(capsys.readouterr().out)
    assert benchmark_payload["iterations"] == 2

    assert main(["list-plugins"]) == 0
    plugins_payload = json.loads(capsys.readouterr().out)
    assert plugins_payload == []
