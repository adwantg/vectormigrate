from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from vectormigrate.registry import SQLiteRegistry
from vectormigrate.reporting import build_migration_report


def export_run_artifact_bundle(
    registry: SQLiteRegistry,
    plan_id: str,
    output_dir: str | Path,
    extra_sections: dict[str, Any] | None = None,
) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    report = build_migration_report(registry, plan_id)
    manifest: dict[str, Any] = {
        "format_version": "1.0",
        "plan_id": plan_id,
        "files": {
            "report": "migration_report.json",
            "manifest": "artifact_manifest.json",
        },
        "extra_sections": sorted((extra_sections or {}).keys()),
    }

    (path / "migration_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if extra_sections:
        for key, payload in extra_sections.items():
            file_name = f"{key}.json"
            (path / file_name).write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            manifest["files"][key] = file_name

    manifest_path = path / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def validate_artifact_bundle(manifest_path: str | Path) -> dict[str, Any]:
    path = Path(manifest_path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required_top_level = {"format_version", "plan_id", "files", "extra_sections"}
    missing_top_level = sorted(required_top_level - set(manifest))
    if missing_top_level:
        raise ValueError(f"artifact manifest missing keys: {missing_top_level}")

    files = manifest["files"]
    if not isinstance(files, dict):
        raise ValueError("artifact manifest 'files' entry must be an object")

    resolved_files: dict[str, str] = {}
    for key, relative_path in files.items():
        file_path = path.parent / relative_path
        if not file_path.exists():
            raise ValueError(f"artifact manifest file missing on disk: {relative_path}")
        json.loads(file_path.read_text(encoding="utf-8"))
        resolved_files[key] = str(file_path)

    return {
        "format_version": manifest["format_version"],
        "plan_id": manifest["plan_id"],
        "resolved_files": resolved_files,
        "extra_sections": list(manifest["extra_sections"]),
    }
