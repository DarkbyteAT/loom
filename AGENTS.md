# AGENTS.md

## Commands

```bash
source scripts/enable-venv.sh
uv run ruff check loom/
uv run pyright loom/
uv run pytest tests/
make all
```

## Critical Rules

- Python 3.11+
- Plain `def test_*` functions, Given-When-Then structure
