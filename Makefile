.PHONY: lint format format-check fix typecheck test all

lint:
	uv run ruff check loom/

format:
	uv run ruff format loom/

format-check:
	uv run ruff format --check loom/

fix:
	uv run ruff check --fix loom/

typecheck:
	uv run pyright loom/

test:
	uv run pytest tests/ -v || [ $$? -eq 5 ]

all: format-check lint typecheck test
