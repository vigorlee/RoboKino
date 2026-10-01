#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

TASK="$1"
shift

if [[ -n "${CONDA_PREFIX:-}" ]]; then
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
        source "${CONDA_BASE}/etc/profile.d/conda.sh"
    fi

    while [[ -n "${CONDA_PREFIX:-}" ]]; do
        conda deactivate || break
    done
fi

cleanup() {
    kill "${rviz_pid:-}" "${roscore_pid:-}" 2>/dev/null || true
}
trap cleanup EXIT

roscore &
roscore_pid=$!
sleep 3

rviz -d "${SCRIPT_DIR}/rviz/fluxbisim.rviz" &
rviz_pid=$!

"${PYTHON_BIN}" -u "${SCRIPT_DIR}/evaluate.py" --task "${TASK}" "$@"
