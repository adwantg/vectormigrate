from __future__ import annotations

import json
import subprocess
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PLUGIN_FIXTURE = ROOT / "tests/fixtures/demo_plugin"


def _venv_python(venv_dir: Path) -> Path:
    return venv_dir / "bin/python"


def _run(command: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=cwd or ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def test_plugin_entry_points_load_from_installed_package(tmp_path: Path) -> None:
    venv_dir = tmp_path / "plugin-venv"
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python_bin = _venv_python(venv_dir)

    _run([str(python_bin), "-m", "pip", "install", "-e", str(ROOT)])
    _run([str(python_bin), "-m", "pip", "install", "-e", str(PLUGIN_FIXTURE)])

    script = """
import json
from vectormigrate.plugins import PluginRegistry

registry = PluginRegistry()
plugins = registry.load_entry_point_plugins()
payload = {
    "plugin_names": [plugin.name for plugin in plugins],
    "factory_result": registry.get("demo-plugin").factory(),
}
print(json.dumps(payload))
"""
    result = _run([str(python_bin), "-c", script])
    payload = json.loads(result.stdout)

    assert "demo-plugin" in payload["plugin_names"]
    assert payload["factory_result"] == "demo-plugin-loaded"
