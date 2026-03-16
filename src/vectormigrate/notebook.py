from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast


def execute_notebook_smoke(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    notebook_path = Path(path)
    root = Path(project_root)
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    globals_dict: dict[str, Any] = {
        "__name__": "__notebook_smoke__",
    }
    old_cwd = Path.cwd()
    previous_path = list(sys.path)
    try:
        sys.path.insert(0, str(root / "src"))
        os.chdir(root)
        for cell in payload.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            source = "".join(cell.get("source", []))
            exec(compile(source, str(notebook_path), "exec"), globals_dict, globals_dict)
    finally:
        os.chdir(old_cwd)
        sys.path[:] = previous_path
    return dict(MappingProxyType(globals_dict))


def execute_notebook_subprocess_smoke(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    notebook_path = Path(path)
    root = Path(project_root)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        output_path = Path(handle.name)

    script = """
import json
import os
import sys
from pathlib import Path

notebook_path = Path(sys.argv[1])
root = Path(sys.argv[2])
output_path = Path(sys.argv[3])
payload = json.loads(notebook_path.read_text(encoding='utf-8'))
globals_dict = {'__name__': '__notebook_smoke__'}
os.chdir(root)
sys.path.insert(0, str(root / 'src'))
executed_code_cells = 0
for cell in payload.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = ''.join(cell.get('source', []))
    executed_code_cells += 1
    exec(compile(source, str(notebook_path), 'exec'), globals_dict, globals_dict)
summary = {
    'executed_code_cells': executed_code_cells,
    'defined_names': sorted(key for key in globals_dict if not key.startswith('__')),
}
output_path.write_text(json.dumps(summary), encoding='utf-8')
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    try:
        subprocess.run(
            [sys.executable, "-c", script, str(notebook_path), str(root), str(output_path)],
            check=True,
            text=True,
            cwd=root,
            env=env,
            capture_output=True,
        )
        return cast(dict[str, Any], json.loads(output_path.read_text(encoding="utf-8")))
    finally:
        output_path.unlink(missing_ok=True)
