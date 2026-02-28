#!/usr/bin/env bash
# Check code quality without modifying files.
# Exits with a non-zero status if any check fails.
set -euo pipefail

cd "$(dirname "$0")/.."

FAILED=0

echo "=== Checking formatting (black) ==="
if uv run black --check backend/ main.py; then
    echo "black: OK"
else
    echo "black: FAILED — run scripts/format.sh to fix"
    FAILED=1
fi

if [ "$FAILED" -ne 0 ]; then
    echo ""
    echo "Quality checks failed. Run scripts/format.sh to auto-fix formatting."
    exit 1
fi

echo ""
echo "All quality checks passed."
