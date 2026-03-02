#!/usr/bin/env bash
# Author: @laitifranz
set -o errexit
set -o nounset
set -o pipefail

if [[ "${TRACE-0}" == "1" ]]; then
    set -o xtrace
fi

print_usage() {
    cat <<'USAGE' >&2
Usage:
  bash scripts/schedule_python.sh path/to/script.py [arg1 arg2 ...]

Examples:
  # run a module inside the repository
  bash scripts/schedule_python.sh src/pipelines/zero_shot/runner.py --config configs/zero_shot.yaml

  # run an ad-hoc script with arbitrary arguments
  bash scripts/schedule_python.sh tools/debug.py --flag foo --limit 5

Notes:
  • The helper activates .venv in the repo root and sets PYTHONPATH to the project root.
  • All arguments after the script name are forwarded verbatim to python.
  • Set TRACE=1 for bash xtrace output.
USAGE
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    print_usage
    exit 0
fi

cd "$(dirname "$0")"
while [ "$(find . -maxdepth 1 -name pyproject.toml | wc -l)" -ne 1 ]; do cd ..; done

source "$(pwd)"/.venv/bin/activate
PYTHONPATH=$(pwd) python "$@"