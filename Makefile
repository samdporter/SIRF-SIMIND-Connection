# Makefile for SIRF-SIMIND-Connection Testing and Development
# 
# This Makefile provides convenient commands for common development and testing tasks.
# Usage: make <target>
#
# Main targets:
#   help            - Show this help message
#   install         - Install package in development mode
#   test            - Run all tests
#   test-quick      - Run quick tests only
#   test-unit       - Run unit tests
#   test-integration - Run integration tests
#   coverage        - Run tests with coverage report
#   lint            - Run code quality checks
#   format          - Format code with black
#   clean           - Clean up temporary files
#   docs            - Build documentation
#   validate        - Run installation validation
#   examples        - Run all example scripts
#   benchmark       - Run performance benchmarks

.PHONY: help install test test-quick test-unit test-integration coverage lint format clean docs validate examples benchmark data docker-build docker-dev docker-shell docker-test docker-clean

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Python and pip executables
PYTHON := python3
PIP := pip3

# Test directories and files
TEST_DIR := tests
EXAMPLE_DIR := examples
SRC_DIR := sirf_simind_connection
DOCS_DIR := docs

# Coverage settings
COVERAGE_MIN := 80
COVERAGE_REPORT_DIR := htmlcov

help: ## Show this help message
	@echo "$(BLUE)SIRF-SIMIND-Connection Development Makefile$(RESET)"
	@echo "=============================================="
	@echo
	@echo "$(GREEN)Python Development Commands:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | grep -v '^docker-' | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo
	@echo "$(GREEN)Docker Commands:$(RESET)"
	@grep -E '^docker-[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-25s$(RESET) %s\n", $$1, $$2}'
	@echo
	@echo "$(GREEN)Quick Start:$(RESET)"
	@echo "  $(YELLOW)Native Python:$(RESET)"
	@echo "    make install           # Install package for development"
	@echo "    make test              # Run native Python tests"
	@echo "    make validate          # Validate native installation"
	@echo
	@echo "  $(YELLOW)Docker (includes SIRF):$(RESET)"
	@echo "    make docker-dev        # Start Docker development environment"
	@echo "    make docker-test       # Run tests in Docker"
	@echo "    make docker-validate   # Validate Docker installation"

# =============================================================================
# Python Development Commands
# =============================================================================

install: ## Install package in development mode with all dependencies
	@echo "$(GREEN)Installing package in development mode...$(RESET)"
	$(PIP) install -e ".[dev]"
	@echo "$(GREEN)Installation complete!$(RESET)"

install-deps: ## Install only dependencies without the package
	@echo "$(GREEN)Installing dependencies...$(RESET)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov pytest-mock black flake8 mypy
	@echo "$(GREEN)Dependencies installed!$(RESET)"

test: ## Run all tests
	@echo "$(GREEN)Running all tests...$(RESET)"
	pytest $(TEST_DIR) -v

test-quick: ## Run quick tests only (exclude slow tests)
	@echo "$(GREEN)Running quick tests...$(RESET)"
	pytest $(TEST_DIR) -v -m "not slow and not requires_simind"

test-unit: ## Run unit tests only
	@echo "$(GREEN)Running unit tests...$(RESET)"
	pytest $(TEST_DIR) -v -m "unit and not requires_simind"

test-integration: ## Run integration tests only
	@echo "$(GREEN)Running integration tests...$(RESET)"
	pytest $(TEST_DIR) -v -m "integration and not requires_simind"

test-simind: ## Run tests that require SIMIND (slow)
	@echo "$(YELLOW)Running SIMIND-dependent tests (requires SIMIND installation)...$(RESET)"
	pytest $(TEST_DIR) -v -m "requires_simind"

test-parallel: ## Run tests in parallel
	@echo "$(GREEN)Running tests in parallel...$(RESET)"
	pytest $(TEST_DIR) -v -n auto

coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(RESET)"
	pytest $(TEST_DIR) \
		--cov=$(SRC_DIR) \
		--cov-report=html:$(COVERAGE_REPORT_DIR) \
		--cov-report=term-missing \
		--cov-report=xml \
		--cov-fail-under=$(COVERAGE_MIN)
	@echo "$(GREEN)Coverage report generated in $(COVERAGE_REPORT_DIR)/$(RESET)"

coverage-html: ## Generate HTML coverage report and open in browser
	@make coverage
	@echo "$(GREEN)Opening coverage report in browser...$(RESET)"
	@if command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(COVERAGE_REPORT_DIR)/index.html; \
	elif command -v open >/dev/null 2>&1; then \
		open $(COVERAGE_REPORT_DIR)/index.html; \
	else \
		echo "$(YELLOW)Please open $(COVERAGE_REPORT_DIR)/index.html manually$(RESET)"; \
	fi

lint: ## Run code quality checks (flake8, mypy)
	@echo "$(GREEN)Running code quality checks...$(RESET)"
	@echo "$(BLUE)Running flake8...$(RESET)"
	flake8 $(SRC_DIR) $(TEST_DIR) --count --statistics
	@echo "$(BLUE)Running mypy...$(RESET)"
	mypy $(SRC_DIR) --ignore-missing-imports

format: ## Format code with black and isort
	@echo "$(GREEN)Formatting code...$(RESET)"
	black $(SRC_DIR) $(TEST_DIR) $(EXAMPLE_DIR)
	isort $(SRC_DIR) $(TEST_DIR) $(EXAMPLE_DIR)
	@echo "$(GREEN)Code formatting complete!$(RESET)"

format-check: ## Check if code formatting is correct
	@echo "$(GREEN)Checking code formatting...$(RESET)"
	black --check --diff $(SRC_DIR) $(TEST_DIR) $(EXAMPLE_DIR)
	isort --check-only --diff $(SRC_DIR) $(TEST_DIR) $(EXAMPLE_DIR)

type-check: ## Run type checking with mypy
	@echo "$(GREEN)Running type checks...$(RESET)"
	mypy $(SRC_DIR) --ignore-missing-imports

security: ## Run security checks with bandit
	@echo "$(GREEN)Running security checks...$(RESET)"
	bandit -r $(SRC_DIR) -f json -o security-report.json || true
	bandit -r $(SRC_DIR)

quality: format-check lint type-check security ## Run all code quality checks

docs: ## Build documentation
	@echo "$(GREEN)Building documentation...$(RESET)"
	@if [ -d "$(DOCS_DIR)" ]; then \
		cd $(DOCS_DIR) && make html; \
		echo "$(GREEN)Documentation built in $(DOCS_DIR)/_build/html/$(RESET)"; \
	else \
		echo "$(YELLOW)Documentation directory not found$(RESET)"; \
	fi

docs-serve: ## Build and serve documentation locally
	@make docs
	@echo "$(GREEN)Serving documentation on http://localhost:8000$(RESET)"
	@cd $(DOCS_DIR)/_build/html && $(PYTHON) -m http.server 8000

validate: ## Run installation validation
	@echo "$(GREEN)Running installation validation...$(RESET)"
	$(PYTHON) scripts/validate_installation.py

validate-quick: ## Run quick installation validation
	@echo "$(GREEN)Running quick validation...$(RESET)"
	$(PYTHON) scripts/validate_installation.py --quick

validate-full: ## Run comprehensive installation validation
	@echo "$(GREEN)Running full validation...$(RESET)"
	$(PYTHON) scripts/validate_installation.py --full

examples: ## Run all example scripts (syntax check only)
	@echo "$(GREEN)Validating example scripts...$(RESET)"
	@for example in $(EXAMPLE_DIR)/*.py; do \
		echo "$(BLUE)Checking $$example...$(RESET)"; \
		$(PYTHON) -m py_compile $$example && echo "$(GREEN)✓ $$example$(RESET)" || echo "$(RED)✗ $$example$(RESET)"; \
	done

examples-run: ## Run all example scripts (actual execution)
	@echo "$(YELLOW)Running example scripts (requires SIRF and SIMIND)...$(RESET)"
	@for example in $(EXAMPLE_DIR)/*.py; do \
		echo "$(BLUE)Running $$example...$(RESET)"; \
		$(PYTHON) $$example || echo "$(RED)Failed: $$example$(RESET)"; \
	done

benchmark: ## Run performance benchmarks
	@echo "$(GREEN)Running performance benchmarks...$(RESET)"
	pytest $(TEST_DIR) -v -m "slow" --benchmark-only || echo "$(YELLOW)Benchmarking requires pytest-benchmark$(RESET)"

profile: ## Run tests with profiling
	@echo "$(GREEN)Running tests with profiling...$(RESET)"
	pytest $(TEST_DIR) --profile || echo "$(YELLOW)Profiling requires pytest-profiling$(RESET)"

data: ## Generate test data
	@echo "$(GREEN)Generating test data...$(RESET)"
	$(PYTHON) scripts/generate_test_data.py --output-dir test_data

data-clean: ## Clean generated test data
	@echo "$(GREEN)Cleaning test data...$(RESET)"
	rm -rf test_data/

clean: ## Clean up temporary files and caches
	@echo "$(GREEN)Cleaning up temporary files...$(RESET)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + || true
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf $(COVERAGE_REPORT_DIR)/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .tox/
	rm -rf validation_*/
	rm -f security-report.json
	rm -f coverage.xml
	@echo "$(GREEN)Cleanup complete!$(RESET)"

clean-all: clean data-clean ## Clean everything including test data
	@echo "$(GREEN)Complete cleanup finished!$(RESET)"

build: ## Build package for distribution
	@echo "$(GREEN)Building package...$(RESET)"
	$(PYTHON) -m build
	@echo "$(GREEN)Package built in dist/$(RESET)"

build-check: ## Check built package
	@make build
	@echo "$(GREEN)Checking built package...$(RESET)"
	twine check dist/*

release-test: ## Upload to test PyPI
	@make build-check
	@echo "$(GREEN)Uploading to test PyPI...$(RESET)"
	twine upload --repository testpypi dist/*

release: ## Upload to PyPI
	@make build-check
	@echo "$(RED)Uploading to PyPI...$(RESET)"
	twine upload dist/*

pre-commit: ## Run pre-commit checks
	@echo "$(GREEN)Running pre-commit checks...$(RESET)"
	@make format-check
	@make lint
	@make test-quick
	@echo "$(GREEN)Pre-commit checks passed!$(RESET)"

ci: ## Run CI pipeline locally
	@echo "$(GREEN)Running CI pipeline locally...$(RESET)"
	@make install
	@make quality
	@make test
	@make coverage
	@make validate-quick
	@echo "$(GREEN)CI pipeline completed!$(RESET)"

dev-setup: ## Set up development environment
	@echo "$(GREEN)Setting up development environment...$(RESET)"
	@make install
	@make data
	@echo "$(GREEN)Development environment ready!$(RESET)"

install-hooks: ## Install git pre-commit hooks
	@echo "$(GREEN)Installing git hooks...$(RESET)"
	@echo "#!/bin/bash" > .git/hooks/pre-commit
	@echo "# Pre-commit hook for SIRF-SIMIND-Connection" >> .git/hooks/pre-commit
	@echo "echo \"Running pre-commit checks...\"" >> .git/hooks/pre-commit
	@echo "make pre-commit" >> .git/hooks/pre-commit
	@chmod +x .git/hooks/pre-commit
	@echo "$(GREEN)Git hooks installed!$(RESET)"

debug-env: ## Show environment information
	@echo "$(GREEN)Environment Information:$(RESET)"
	@echo "Python version: $$($(PYTHON) --version)"
	@echo "Python executable: $$(which $(PYTHON))"
	@echo "Pip version: $$($(PIP) --version)"
	@echo "Current directory: $$(pwd)"
	@echo "Git branch: $$(git branch --show-current 2>/dev/null || echo 'Not a git repository')"
	@echo "Git status: $$(git status --porcelain 2>/dev/null | wc -l) modified files"
	@echo "Installed packages:"
	@$(PIP) list | grep -E "(sirf|simind|pytest|numpy|scipy)" || echo "No relevant packages found"

debug-deps: ## Check dependency status
	@echo "$(GREEN)Dependency Status:$(RESET)"
	@for dep in numpy scipy matplotlib yaml pytest; do \
		$(PYTHON) -c "import $$dep; print('✓ $$dep:', $$dep.__version__)" 2>/dev/null || echo "✗ $$dep: not available"; \
	done
	@$(PYTHON) -c "import sirf; print('✓ SIRF available')" 2>/dev/null || echo "✗ SIRF: not available"
	@$(PYTHON) -c "import sirf_simind_connection; print('✓ Package available')" 2>/dev/null || echo "✗ Package: not available"
	@which simind >/dev/null 2>&1 && echo "✓ SIMIND: available" || echo "✗ SIMIND: not in PATH"

watch-tests: ## Watch for file changes and run tests automatically
	@echo "$(GREEN)Watching for changes (requires entr)...$(RESET)"
	@find $(SRC_DIR) $(TEST_DIR) -name "*.py" | entr -c make test-quick

memory-test: ## Run memory usage test
	@echo "$(GREEN)Running memory usage test...$(RESET)"
	$(PYTHON) -c "import psutil; import sys; process = psutil.Process(); print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB'); import sirf_simind_connection; print(f'After import: {process.memory_info().rss / 1024 / 1024:.1f} MB')" 2>/dev/null || echo "Import test failed"

readme-toc: ## Generate table of contents for README
	@echo "$(GREEN)Generating README table of contents...$(RESET)"
	@$(PYTHON) -c "import re; content = open('README.md').read(); headers = re.findall(r'^(#+)\s+(.+)$$', content, re.MULTILINE); [print('  ' * (len(level) - 1) + f'- [{title}](#{title.lower().replace(\" \", \"-\").replace(\"/\", \"\").replace(\"(\", \"\").replace(\")\", \"\")})') for level, title in headers]"

info: ## Show project information
	@echo "$(BLUE)SIRF-SIMIND-Connection Project Information$(RESET)"
	@echo "============================================="
	@echo "Project: SIRF-SIMIND-Connection"
	@echo "Description: Python wrapper for SIRF and SIMIND integration"
	@echo "Repository: https://github.com/samdporter/SIRF-SIMIND-Connection"
	@echo ""
	@echo "$(GREEN)Quick Start:$(RESET)"
	@echo "  make dev-setup        # Set up development environment"
	@echo "  make test             # Run tests"
	@echo "  make validate         # Validate installation"
	@echo "  make docker-dev       # Start Docker development environment"
	@echo ""
	@echo "$(GREEN)Common Commands:$(RESET)"
	@echo "  make help             # Show all available commands"
	@echo "  make ci               # Run full CI pipeline"
	@echo "  make pre-commit       # Run pre-commit checks"

status: ## Show project status
	@echo "$(GREEN)Project Status:$(RESET)"
	@echo "Git status: $$(git status --porcelain | wc -l) modified files"
	@echo "Test status: $$(make test-quick >/dev/null 2>&1 && echo '✓ Passing' || echo '✗ Failing')"
	@echo "Dependencies: $$(make debug-deps 2>/dev/null | grep '✓' | wc -l) available"

verify-install: ## Verify package can be installed and imported
	@echo "$(GREEN)Verifying package installation...$(RESET)"
	@$(PYTHON) -c "import sys; import subprocess; import tempfile; import os; print('Creating temporary environment...'); exec(open('verify_install.py').read())" 2>/dev/null || echo "Verification script not found"

# =============================================================================
# Docker Commands
# =============================================================================

docker-build: ## Build all Docker images
	@echo "$(GREEN)Building all Docker images...$(RESET)"
	docker-compose build
	@echo "$(GREEN)✅ Docker images built$(RESET)"

docker-build-dev: ## Build development image only
	@echo "$(GREEN)Building development image...$(RESET)"
	docker-compose build sirf-simind-dev
	@echo "$(GREEN)✅ Development image built$(RESET)"

docker-build-test: ## Build test image only
	@echo "$(GREEN)Building test image...$(RESET)"
	docker-compose build test-unit
	@echo "$(GREEN)✅ Test image built$(RESET)"

docker-dev: ## Start development environment in background
	@echo "$(GREEN)Starting Docker development environment...$(RESET)"
	docker-compose --profile dev up -d
	@echo "$(GREEN)✅ Development environment started$(RESET)"
	@echo "$(YELLOW)Services available:$(RESET)"
	@echo "  Jupyter Lab: http://localhost:8888"
	@echo "  Shell access: make docker-shell"
	@echo "  Logs: make docker-logs"

docker-dev-fg: ## Start development environment in foreground
	@echo "$(GREEN)Starting Docker development environment (foreground)...$(RESET)"
	docker-compose --profile dev up

docker-shell: ## Open interactive shell in development container
	@echo "$(GREEN)Opening shell in development container...$(RESET)"
	@if docker-compose ps sirf-simind-dev | grep -q "Up"; then \
		docker-compose exec sirf-simind-dev bash; \
	else \
		echo "$(YELLOW)Starting development container...$(RESET)"; \
		docker-compose --profile debug run --rm shell; \
	fi

docker-root-shell: ## Open root shell in development container
	@echo "$(GREEN)Opening root shell in development container...$(RESET)"
	@if docker-compose ps sirf-simind-dev | grep -q "Up"; then \
		docker-compose exec --user root sirf-simind-dev bash; \
	else \
		echo "$(YELLOW)Starting development container as root...$(RESET)"; \
		docker-compose --profile debug run --rm --user root shell; \
	fi

docker-validate: ## Quick Docker validation check
	@echo "$(GREEN)Running Docker validation...$(RESET)"
	docker-compose run --rm validate
	@echo "$(GREEN)✅ Docker validation completed$(RESET)"

docker-test: ## Run unit tests in Docker (no SIMIND required)
	@echo "$(GREEN)Running Docker unit tests...$(RESET)"
	docker-compose run --rm test-unit
	@echo "$(GREEN)✅ Docker unit tests completed$(RESET)"

docker-test-integration: ## Run integration tests in Docker (requires SIMIND)
	@echo "$(GREEN)Running Docker integration tests...$(RESET)"
	@if docker-compose exec sirf-simind-dev command -v simind >/dev/null 2>&1; then \
		docker-compose run --rm test-integration; \
		echo "$(GREEN)✅ Integration tests completed$(RESET)"; \
	else \
		echo "$(RED)❌ SIMIND not installed$(RESET)"; \
		echo "$(YELLOW)Run 'make docker-install-simind' first$(RESET)"; \
		exit 1; \
	fi

docker-test-full: ## Run complete test suite in Docker
	@echo "$(GREEN)Running complete Docker test suite...$(RESET)"
	docker-compose --profile test up --abort-on-container-exit
	@echo "$(GREEN)✅ Complete Docker test suite finished$(RESET)"

docker-lint: ## Run code quality checks in Docker
	@echo "$(GREEN)Running Docker lint checks...$(RESET)"
	docker-compose run --rm lint
	@echo "$(GREEN)✅ Docker lint checks completed$(RESET)"

docker-benchmark: ## Run performance benchmarks in Docker
	@echo "$(GREEN)Running Docker benchmarks...$(RESET)"
	docker-compose --profile benchmark up --abort-on-container-exit
	@echo "$(GREEN)✅ Docker benchmarks completed$(RESET)"
	@echo "$(YELLOW)Results saved to: ./benchmark-results/$(RESET)"

docker-docs: ## Build and serve documentation in Docker
	@echo "$(GREEN)Building Docker documentation...$(RESET)"
	docker-compose --profile docs up -d
	@echo "$(GREEN)✅ Documentation server started$(RESET)"
	@echo "$(YELLOW)Documentation available at: http://localhost:8000$(RESET)"

docker-jupyter: ## Start Jupyter Lab in Docker
	@echo "$(GREEN)Starting Docker Jupyter Lab...$(RESET)"
	docker-compose --profile jupyter up -d
	@echo "$(GREEN)✅ Jupyter Lab started$(RESET)"
	@echo "$(YELLOW)Jupyter Lab available at: http://localhost:8888$(RESET)"

docker-examples: ## Validate examples in Docker
	@echo "$(GREEN)Validating Docker examples...$(RESET)"
	docker-compose --profile examples up --abort-on-container-exit
	@echo "$(GREEN)✅ Docker example validation completed$(RESET)"

docker-install-simind: ## Install SIMIND in Docker container interactively
	@echo "$(BLUE)SIMIND Installation in Docker$(RESET)"
	@echo "============================="
	@echo "$(YELLOW)Prerequisites:$(RESET)"
	@echo "1. Download SIMIND from: https://simind.blogg.lu.se/downloads/"
	@echo "2. Have the archive file ready"
	@echo ""
	@echo "$(YELLOW)Starting development container if not running...$(RESET)"
	@docker-compose --profile dev up -d || true
	@echo "$(YELLOW)Please enter the path to your SIMIND archive:$(RESET)"
	@read -p "Archive path: " archive; \
	if [ -f "$$archive" ]; then \
		echo "$(BLUE)Copying archive to container...$(RESET)"; \
		docker cp "$$archive" sirf-simind-dev:/tmp/simind_archive; \
		echo "$(BLUE)Installing SIMIND...$(RESET)"; \
		docker-compose exec sirf-simind-dev install_simind.sh /tmp/simind_archive; \
		echo "$(GREEN)✅ SIMIND installation completed$(RESET)"; \
	else \
		echo "$(RED)❌ Archive file not found: $$archive$(RESET)"; \
		exit 1; \
	fi

docker-validate-simind: ## Validate SIMIND installation in Docker
	@echo "$(GREEN)Validating SIMIND installation...$(RESET)"
	docker-compose exec sirf-simind-dev install_simind.sh --validate-only

docker-stop: ## Stop all Docker services
	@echo "$(GREEN)Stopping Docker services...$(RESET)"
	docker-compose down
	@echo "$(GREEN)✅ All Docker services stopped$(RESET)"

docker-restart: ## Restart development environment
	@echo "$(GREEN)Restarting Docker development environment...$(RESET)"
	docker-compose restart sirf-simind-dev
	@echo "$(GREEN)✅ Development environment restarted$(RESET)"

docker-clean: ## Clean Docker containers and volumes
	@echo "$(GREEN)Cleaning Docker environment...$(RESET)"
	docker-compose down -v
	docker system prune -f
	@echo "$(GREEN)✅ Docker cleanup completed$(RESET)"

docker-clean-all: ## Remove everything including images
	@echo "$(RED)⚠️  This will remove all containers, volumes, and images$(RESET)"
	@echo "$(YELLOW)Are you sure? Type 'yes' to continue:$(RESET)"
	@read confirm; \
	if [ "$$confirm" = "yes" ]; then \
		echo "$(BLUE)Removing everything...$(RESET)"; \
		docker-compose down -v --rmi all; \
		docker system prune -f -a; \
		echo "$(GREEN)✅ Complete Docker cleanup finished$(RESET)"; \
	else \
		echo "$(YELLOW)Cleanup cancelled$(RESET)"; \
	fi

docker-clean-results: ## Clean test and benchmark result directories
	@echo "$(GREEN)Cleaning Docker result directories...$(RESET)"
	@if [ -d "test-results" ]; then rm -rf test-results/*; fi
	@if [ -d "benchmark-results" ]; then rm -rf benchmark-results/*; fi
	@if [ -d "lint-results" ]; then rm -rf lint-results/*; fi
	@echo "$(GREEN)✅ Result directories cleaned$(RESET)"

docker-logs: ## Show logs for all Docker services
	@echo "$(GREEN)Docker service logs:$(RESET)"
	docker-compose logs --tail=50

docker-logs-follow: ## Follow logs for all Docker services
	@echo "$(GREEN)Following Docker logs (Ctrl+C to stop):$(RESET)"
	docker-compose logs -f

docker-status: ## Show Docker environment status
	@echo "$(GREEN)Docker Environment Status:$(RESET)"
	@echo "========================="
	@echo "Docker version: $$(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker Compose version: $$(docker-compose --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Running services:"
	@docker-compose ps 2>/dev/null || echo "No services running"
	@echo ""
	@echo "Available images:"
	@docker images | grep sirf-simind-connection || echo "No images built yet"

docker-ci: ## Run complete CI pipeline in Docker
	@echo "$(GREEN)Running Docker CI pipeline...$(RESET)"
	@make docker-build
	@make docker-validate
	@make docker-test
	@make docker-lint
	@make docker-benchmark
	@echo "$(GREEN)✅ Docker CI pipeline completed!$(RESET)"

docker-quick: docker-validate ## Quick Docker setup and validation

docker-dev-setup: ## Complete Docker development setup
	@echo "$(GREEN)Setting up Docker development environment...$(RESET)"
	@make docker-build-dev
	@make docker-dev
	@make docker-validate
	@echo "$(GREEN)✅ Docker development environment ready!$(RESET)"
	@echo "$(YELLOW)Next steps:$(RESET)"
	@echo "  make docker-shell      # Open development shell"
	@echo "  make docker-test       # Run tests"
	@echo "  make docker-jupyter    # Start Jupyter Lab"

docker-reset: ## Reset Docker development environment
	@echo "$(GREEN)Resetting Docker development environment...$(RESET)"
	@make docker-clean
	@make docker-build
	@make docker-dev
	@echo "$(GREEN)✅ Docker development environment reset$(RESET)"

list: ## List all available targets
	@make help