from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = [
    "example_registry_and_plan.py",
    "example_compatibility.py",
    "example_backend_requests.py",
    "example_artifacts_and_telemetry.py",
    "example_assets_and_notebook.py",
]


def test_example_scripts_run_successfully() -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src")

    for script_name in EXAMPLES:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "examples" / script_name)],
            check=True,
            capture_output=True,
            text=True,
            cwd=ROOT,
            env=env,
        )
        payload = json.loads(completed.stdout)
        assert isinstance(payload, dict)
