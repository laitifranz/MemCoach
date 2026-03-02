SHELL := /bin/bash

help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean-logs: ## Clean logs
	rm -rf logs/**

setup-pre-commit: ## One-time setup: install pre-commit globally with uv + install git hooks
	uv tool install pre-commit --with pre-commit-uv --force-reinstall
	pre-commit install --install-hooks  # --install-hooks also seeds the environments
	pre-commit autoupdate

run-pre-commit: ## Run all pre-commit hooks on the entire codebase
	pre-commit run --all-files

run-tests: ## Run tests
	uv add pytest
	PYTHONPATH=$(CURDIR) uv run pytest tests

check: run-pre-commit    ## Alias