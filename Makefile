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

.PHONY: help install test test-quick test-unit test-integration coverage lint format clean docs validate examples benchmark data

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
	@echo "$(GREEN)Available targets:$(RESET)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(BLUE)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo
	@echo "$(GREEN)Examples:$(RESET)"
	@echo "  make install        # Install package for development"
	@echo "  make test           # Run all tests"
	@echo "  make coverage       # Generate coverage report"
	@echo "  make lint           # Check code quality"
	@echo "  make clean          # Clean temporary files"

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

# Git hooks
install-hooks: ## Install git pre-commit hooks
	@echo "$(GREEN)Installing git hooks...$(RESET)"
	@cat > .git/hooks/pre-commit << 'EOF'
#!/bin/bash
# Pre-commit hook for SIRF-SIMIND-Connection
echo "Running pre-commit checks..."
make pre-commit
EOF
	@chmod +x .git/hooks/pre-commit
	@echo "$(GREEN)Git hooks installed!$(RESET)"

# Debug and troubleshooting
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

# Performance monitoring
memory-test: ## Run memory usage test
	@echo "$(GREEN)Running memory usage test...$(RESET)"
	$(PYTHON) -c "
import psutil
import sys
process = psutil.Process()
print(f'Memory usage: {process.memory_info().rss / 1024 / 1024:.1f} MB')
try:
    import sirf_simind_connection
    print(f'After import: {process.memory_info().rss / 1024 / 1024:.1f} MB')
except ImportError as e:
    print(f'Import failed: {e}')
"

# Documentation helpers
readme-toc: ## Generate table of contents for README
	@echo "$(GREEN)Generating README table of contents...$(RESET)"
	@$(PYTHON) -c "
import re
with open('README.md', 'r') as f:
    content = f.read()
headers = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
toc = []
for level, title in headers:
    indent = '  ' * (len(level) - 1)
    link = title.lower().replace(' ', '-').replace('/', '').replace('(', '').replace(')', '')
    toc.append(f'{indent}- [{title}](#{link})')
print('\\n'.join(toc))
"

# Info targets
info: ## Show project information
	@echo "$(BLUE)SIRF-SIMIND-Connection Project Information$(RESET)"
	@echo "============================================="
	@echo "Project: SIRF-SIMIND-Connection"
	@echo "Description: Python wrapper for SIRF and SIMIND integration"
	@echo "Repository: https://github.com/samdporter/SIRF-SIMIND-Connection"
	@echo ""
	@echo "$(GREEN)Quick Start:$(RESET)"
	@echo "  make dev-setup    # Set up development environment"
	@echo "  make test         # Run tests"
	@echo "  make validate     # Validate installation"
	@echo ""
	@echo "$(GREEN)Common Commands:$(RESET)"
	@echo "  make help         # Show all available commands"
	@echo "  make ci           # Run full CI pipeline"
	@echo "  make pre-commit   # Run pre-commit checks"

status: ## Show project status
	@echo "$(GREEN)Project Status:$(RESET)"
	@echo "Git status: $$(git status --porcelain | wc -l) modified files"
	@echo "Test status: $$(make test-quick >/dev/null 2>&1 && echo '✓ Passing' || echo '✗ Failing')"
	@echo "Coverage: $$(python -c "import coverage; c = coverage.Coverage(); c.load(); print(f'{c.report():.1f}%')" 2>/dev/null || echo 'Unknown')"
	@echo "Dependencies: $$(make debug-deps 2>/dev/null | grep '✓' | wc -l) available"

# Advanced targets for CI/CD
# Docker-based testing targets
docker-build: ## Build Docker test image
	@echo "$(GREEN)Building Docker test image...$(RESET)"
	docker build -f Dockerfile.test-without-simind -t sirf-simind-connection:test .

docker-build-dev: ## Build Docker development image with SIRF
	@echo "$(GREEN)Building Docker development image...$(RESET)"
	docker build -f Dockerfile.sirf-simind -t sirf-simind-connection:dev .

docker-test: ## Run tests in Docker container (without SIMIND)
	@echo "$(GREEN)Running tests in Docker...$(RESET)"
	docker run --rm -v $(PWD):/workspace sirf-simind-connection:test \
		bash -c "cd /workspace && pytest tests/ -v -m 'not requires_simind' --tb=short"

docker-test-full: ## Run full Docker test suite using docker-compose
	@echo "$(GREEN)Running full Docker test suite...$(RESET)"
	docker-compose up --build --abort-on-container-exit test-no-simind

docker-dev: ## Start interactive development container
	@echo "$(GREEN)Starting Docker development environment...$(RESET)"
	docker-compose up --build sirf-simind-dev

docker-shell: ## Open shell in development container
	@echo "$(GREEN)Opening shell in Docker container...$(RESET)"
	docker run -it --rm -v $(PWD):/home/sirfuser/workspace \
		sirf-simind-connection:dev /bin/bash

docker-benchmark: ## Run performance benchmarks in Docker
	@echo "$(GREEN)Running benchmarks in Docker...$(RESET)"
	docker-compose up --build --abort-on-container-exit benchmark

docker-examples: ## Validate examples in Docker
	@echo "$(GREEN)Validating examples in Docker...$(RESET)"
	docker-compose up --build --abort-on-container-exit examples

docker-clean: ## Clean Docker images and containers
	@echo "$(GREEN)Cleaning Docker images...$(RESET)"
	docker-compose down --rmi all --volumes --remove-orphans || true
	docker rmi sirf-simind-connection:test sirf-simind-connection:dev || true
	docker system prune -f

docker-install-simind: ## Instructions for installing SIMIND in Docker
	@echo "$(BLUE)Installing SIMIND in Docker container:$(RESET)"
	@echo "1. Download SIMIND from https://simind.blogg.lu.se/downloads/"
	@echo "2. Start the development container:"
	@echo "   make docker-dev"
	@echo "3. In another terminal, copy SIMIND archive to container:"
	@echo "   docker cp simind_linux.tar.gz sirf-simind-dev:/tmp/"
	@echo "4. Install SIMIND in the container:"
	@echo "   docker exec sirf-simind-dev /tmp/install_simind.sh /tmp/simind_linux.tar.gz"
	@echo "5. Restart container to apply PATH changes"

# Show Docker-specific status
docker-status: ## Show Docker environment status
	@echo "$(GREEN)Docker Environment Status:$(RESET)"
	@echo "Docker version: $(docker --version 2>/dev/null || echo 'Not installed')"
	@echo "Docker Compose: $(docker-compose --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Available images:"
	@docker images sirf-simind-connection 2>/dev/null || echo "No images built yet"
	@echo ""
	@echo "Running containers:"
	@docker ps --filter "name=sirf-simind" 2>/dev/null || echo "No containers running"

# Package verification
verify-install: ## Verify package can be installed and imported
	@echo "$(GREEN)Verifying package installation...$(RESET)"
	@$(PYTHON) -c "
import sys
import subprocess
import tempfile
import os

# Create temporary environment
print('Creating temporary environment...')
with tempfile.TemporaryDirectory() as tmpdir:
    venv_path = os.path.join(tmpdir, 'test_env')
    subprocess.run([sys.executable, '-m', 'venv', venv_path], check=True)
    
    # Install package
    pip_path = os.path.join(venv_path, 'bin', 'pip') if os.name != 'nt' else os.path.join(venv_path, 'Scripts', 'pip.exe')
    python_path = os.path.join(venv_path, 'bin', 'python') if os.name != 'nt' else os.path.join(venv_path, 'Scripts', 'python.exe')
    
    print('Installing package...')
    subprocess.run([pip_path, 'install', '-e', '.'], check=True)
    
    print('Testing import...')
    result = subprocess.run([python_path, '-c', 'import sirf_simind_connection; print(\"✓ Import successful\")'], 
                          capture_output=True, text=True)
    
    if result.returncode == 0:
        print('✓ Package verification successful')
    else:
        print('✗ Package verification failed')
        print(result.stderr)
        sys.exit(1)
"

# Show makefile targets with descriptions
list: ## List all available targets
	@make help