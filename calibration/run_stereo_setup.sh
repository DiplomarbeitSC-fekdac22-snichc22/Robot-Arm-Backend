#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
BACKEND_SERVICE="robot-arm-backend.service"
DEFAULT_PYTHON="/home/robot/venvs/camera/bin/python"
SERVICE_WAS_ACTIVE=0

if [[ -x "${DEFAULT_PYTHON}" ]]; then
    PYTHON_BIN="${DEFAULT_PYTHON}"
else
    PYTHON_BIN="$(command -v python3)"
fi

restart_backend() {
    if [[ "${SERVICE_WAS_ACTIVE}" -eq 1 ]]; then
        echo "Restarting ${BACKEND_SERVICE}..."
        sudo systemctl start "${BACKEND_SERVICE}"
    fi
}

trap restart_backend EXIT INT TERM

if systemctl list-unit-files "${BACKEND_SERVICE}" --no-legend 2>/dev/null \
    | grep -q "^${BACKEND_SERVICE}"; then
    if systemctl is-active --quiet "${BACKEND_SERVICE}"; then
        SERVICE_WAS_ACTIVE=1
        echo "Stopping ${BACKEND_SERVICE} so calibration can use both cameras..."
        sudo systemctl stop "${BACKEND_SERVICE}"
    fi
fi

cd "${REPOSITORY_DIR}"

"${PYTHON_BIN}" \
    calibration/stereo_auto_setup.py \
    --config calibration/stereo_setup.toml \
    "$@"
