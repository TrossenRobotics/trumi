DOCS_DIR ?= docs
DOCS_BUILD_DIR ?= $(DOCS_DIR)/_build
DOCS_PORT ?= 8000

# Isolated environment for the docs build (only the `docs` dependency group).
export UV_PROJECT_ENVIRONMENT := $(DOCS_BUILD_DIR)/venv

docs:
	uv sync --only-group docs
	uv run --no-sync sphinx-build -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html
.PHONY: docs

docs-strict:
	uv sync --only-group docs
	uv run --no-sync sphinx-build -W --keep-going -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html
.PHONY: docs-strict

docs-serve: docs
	uv run --no-sync python -m http.server $(DOCS_PORT) --directory $(DOCS_BUILD_DIR)/html
.PHONY: docs-serve

docs-clean:
	rm -rf $(DOCS_BUILD_DIR)
.PHONY: docs-clean
