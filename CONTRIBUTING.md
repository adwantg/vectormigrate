# Contributing to vectormigrate

## Developer setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m pytest
```

## Quality bar

Every change should:

- keep the migration workflow modular
- include tests for the new behavior and its obvious edge cases
- update `README.md` when the CLI, public API, or scope changes
- preserve at least 90% coverage

## Useful checks

```bash
ruff format .
ruff check .
mypy src
python -m pytest
python3 scripts/run_live_backend_tests.py
```
