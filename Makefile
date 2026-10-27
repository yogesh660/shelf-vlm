# One entry point for humans, CI and coding agents. Keep targets in sync with CLAUDE.md
# (tests/test_harness.py fails if CLAUDE.md mentions a target that does not exist).

RUN := uv run
DATA ?= data/sku110k

.DEFAULT_GOAL := help
.PHONY: help setup lock lint format typecheck test check schema data data-stats demo clean

help: ## List available targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-12s %s\n", $$1, $$2}'

setup: ## Install the core package and dev tools, enable the pre-commit hook
	uv sync
	$(RUN) pre-commit install

lock: ## Re-resolve dependencies into uv.lock after editing pyproject.toml
	uv lock

lint: ## Ruff lint and format check (no changes)
	$(RUN) ruff check src tests
	$(RUN) ruff format --check src tests

format: ## Auto-fix lint issues and format code
	$(RUN) ruff check --fix src tests
	$(RUN) ruff format src tests

typecheck: ## Mypy (strict) on src/
	$(RUN) mypy

test: ## Run the full pytest suite
	$(RUN) pytest

check: lint typecheck test ## Everything CI runs. Must pass before any commit.

schema: ## Print the JSON schema the VLM must answer in
	$(RUN) shelf schema

data: ## Download and unpack SKU-110K into $(DATA) (about 13.6 GB)
	$(RUN) shelf data download --dest $(DATA)

data-stats: ## Image and box counts per split
	$(RUN) shelf data stats --dest $(DATA)

demo: ## Rows, gaps and compliance on the bundled example, no data or GPU needed
	$(RUN) shelf rows tests/fixtures/annotations_mini.csv shelf_a.jpg
	$(RUN) shelf score --planogram examples/planogram.json --report examples/report.json

clean: ## Remove caches
	rm -rf .pytest_cache .mypy_cache .ruff_cache
