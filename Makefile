.PHONY: sync lint typecheck test build install

sync:
	uv sync

lint:
	uv run ruff check easyupscaler tests

typecheck:
	uv run mypy easyupscaler

test: lint typecheck
	uv run pytest

build:
	uv build

install:
	python -m pip uninstall easyupscaler
	python -m pip install .
