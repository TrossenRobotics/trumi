DOCS_DIR ?= docs
DOCS_BUILD_DIR ?= $(DOCS_DIR)/_build
DOCS_PORT ?= 8000
UV ?= uv

docs:
	$(UV) run --group docs python -m sphinx -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html
	@echo "Docs built: $(DOCS_BUILD_DIR)/html/index.html"
.PHONY: docs

docs-strict:
	$(UV) run --group docs python -m sphinx -W --keep-going -b html $(DOCS_DIR) $(DOCS_BUILD_DIR)/html
.PHONY: docs-strict

docs-serve: docs
	$(UV) run --group docs python -m http.server $(DOCS_PORT) --directory $(DOCS_BUILD_DIR)/html
.PHONY: docs-serve

docs-clean:
	rm -rf $(DOCS_BUILD_DIR)
.PHONY: docs-clean
