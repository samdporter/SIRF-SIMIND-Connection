#!/usr/bin/env bash
set -e

# 1. Apply Ruff lint fixes, including import sorting.
echo "Applying Ruff lint fixes..."
ruff check . --fix

# 2. Format with Ruff.
echo "Formatting with Ruff..."
ruff format .

# 3. Lint check
echo "Running lint check with Ruff..."
ruff check .
