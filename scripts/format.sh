#!/usr/bin/env bash
# Apply black formatting to all Python source files
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Running black formatter..."
uv run black backend/ main.py
echo "Done."
