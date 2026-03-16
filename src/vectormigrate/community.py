from __future__ import annotations

from pathlib import Path

COMMUNITY_GUIDELINES = """# Community Adapter Governance

## Principles

1. Backend adapters must declare capabilities explicitly.
2. New adapters must include unit tests for request compilation and failure handling.
3. Live integration tests should be opt-in and isolated from the default test suite.
4. Public adapter APIs should remain backend-neutral where possible.
5. Benchmark and telemetry outputs should use structured JSON artifacts.
"""


def write_governance_file(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(COMMUNITY_GUIDELINES, encoding="utf-8")
    return output_path
