#!/usr/bin/env bash
# Copyright (c) 2026 vigorlee
# SPDX-License-Identifier: MIT
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: PYTHON_BIN=/path/to/isaac-sim-4.5.0/python.sh bash robokino_benchmark.sh TASK [OPTIONS]

Tasks: close_box, handover_book, pick_place_banana, pull_push_drawer, screw_pitcher_lid
Options are passed to evaluate.py, including --num-episodes, --chunk-size, --save-images.
Requires Isaac Sim 4.5.0, simulation assets, and a compatible ROS 1 environment.
EOF
}

if [[ $# -eq 0 || "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi

case "$1" in
    close_box|handover_book|pick_place_banana|pull_push_drawer|screw_pitcher_lid) ;;
    *) printf 'Unknown task: %s\n' "$1" >&2; usage >&2; exit 2 ;;
esac

if [[ -z "${PYTHON_BIN:-}" ]]; then
    printf 'Set PYTHON_BIN to the Python launcher bundled with Isaac Sim.\n' >&2
    exit 2
fi

for executable in "$PYTHON_BIN" roscore rviz; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Required command is unavailable: %s\n' "$executable" >&2
        exit 127
    fi
done

ROBO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$ROBO_ROOT"
exec bash "$ROBO_ROOT/fluxbisim_benchmark.sh" "$@"
