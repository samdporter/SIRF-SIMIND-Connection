#!/usr/bin/env bash
set -e

# 1. Ruff’s auto‐formatter
echo "Running Ruff auto-formatter..."
ruff format sirf_simind_connection
ruff format examples

# 2. Black formatting
echo "Running Black formatter..."
black .

# 3. Sort imports
echo "Sorting imports with isort..."
isort .

# 4. Remove unused imports & variables
echo "Removing unused imports and variables with autoflake..."
autoflake --in-place \
          --remove-all-unused-imports \
          --remove-unused-variables \
          --recursive .

# 5. Final lint check
echo "Running final lint check with Ruff..."
ruff check ./sirf_simind_connection examples
