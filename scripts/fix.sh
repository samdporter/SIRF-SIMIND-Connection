#!/usr/bin/env bash
set -e

# 1. Remove unused imports & variables
echo "Removing unused imports and variables with autoflake..."
autoflake --in-place \
          --remove-all-unused-imports \
          --remove-unused-variables \
          --recursive .

# 2. Black formatting
echo "Running Black formatter..."
black .

# 3. Sort imports
echo "Sorting imports with isort..."
isort .

# 4. Lint check
echo "Running lint check with Ruff..."
ruff check ./sirf_simind_connection examples