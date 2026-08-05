from __future__ import annotations

import math
import os
import subprocess
import threading
from pathlib import Path
from typing import Any
import subprocess
import threading


KINEMATICS_DIR = Path(
    os.getenv(
        "KINEMATICS_DIR",
        "/home/robot/dev/Kinematics",
    )
)

KINEMATICS_PYTHON = Path(
    os.getenv(
        "KINEMATICS_PYTHON",
        "/home/robot/venvs/Kinematics/bin/python",
    )
)

KINEMATICS_CONFIG_DIR = Path(
    os.getenv(
        "KINEMATICS_CONFIG_DIR",
        str(KINEMATICS_DIR / "configs"),
    )
)

KINEMATICS_RUNNER = Path(
    os.getenv(
        "KINEMATICS_RUNNER",
        str(
            KINEMATICS_DIR
            / "src"
            / "kinematics_hardware_runner.py"
        ),
    )
)

KINEMATICS_START_POSE = os.getenv(
    "KINEMATICS_START_POSE",
    "ready",
)

KINEMATICS_TIMEOUT_SECONDS = float(
    os.getenv(
        "KINEMATICS_TIMEOUT_SECONDS",
        "210",
    )
)

_hardware_lock = threading.Lock()


class KinematicsBridgeError(RuntimeError):
    pass


def _validate_coordinate(
    name: str,
    value: float,
) -> float:
    value = float(value)

    if not math.isfinite(value):
        raise KinematicsBridgeError(
            f"{name} must be finite"
        )

    return value

def run_scan_pose() -> dict[str, object]:
    command = [
        "/home/robot/venvs/Kinematics/bin/robot-arm",
        "pose",
        "sees_everything_mono",
        "--hardware",
        "--from-pose",
        "sees_everything_mono",
    ]

    if not _hardware_lock.acquire(blocking=False):
        raise KinematicsBridgeError(
            "Another robot movement is already running."
        )

    try:
        result = subprocess.run(
            command,
            input="MOVE\nrelease\n",
            text=True,
            capture_output=True,
            cwd="/home/robot/dev/Kinematics",
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise KinematicsBridgeError(
            "Moving to the camera position timed out."
        ) from error
    finally:
        _hardware_lock.release()

    if result.returncode != 0:
        message = (
            result.stderr.strip()
            or result.stdout.strip()
            or f"Kinematics exited with code {result.returncode}"
        )

        raise KinematicsBridgeError(message)

    return {
        "accepted": True,
        "mode": "scan_pose_hardware_run",
        "pose": "sees_everything_mono",
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

def run_hardware_pick(
    *,
    x_mm: float,
    y_mm: float,
    z_mm: float,
) -> dict[str, Any]:
    x_mm = _validate_coordinate("x_mm", x_mm)
    y_mm = _validate_coordinate("y_mm", y_mm)
    z_mm = _validate_coordinate("z_mm", z_mm)

    if not KINEMATICS_DIR.is_dir():
        raise KinematicsBridgeError(
            f"Kinematics directory not found: "
            f"{KINEMATICS_DIR}"
        )

    if not KINEMATICS_PYTHON.is_file():
        raise KinematicsBridgeError(
            f"Kinematics Python not found: "
            f"{KINEMATICS_PYTHON}"
        )

    if not KINEMATICS_CONFIG_DIR.is_dir():
        raise KinematicsBridgeError(
            f"Kinematics config directory not found: "
            f"{KINEMATICS_CONFIG_DIR}"
        )

    if not KINEMATICS_RUNNER.is_file():
        raise KinematicsBridgeError(
            f"Kinematics hardware runner not found: "
            f"{KINEMATICS_RUNNER}"
        )

    if not _hardware_lock.acquire(blocking=False):
        raise KinematicsBridgeError(
            "A Kinematics hardware operation is already running"
        )

    try:
        command = [
            str(KINEMATICS_PYTHON),
            str(KINEMATICS_RUNNER),
            f"{x_mm:.3f}",
            f"{y_mm:.3f}",
            f"{z_mm:.3f}",
            "--start-pose",
            KINEMATICS_START_POSE,
            "--config-dir",
            str(KINEMATICS_CONFIG_DIR),
            "--confirm",
            "MOVE",
        ]

        environment = os.environ.copy()

        kinematics_src = str(
            KINEMATICS_DIR / "src"
        )

        existing_pythonpath = environment.get(
            "PYTHONPATH",
            "",
        )

        environment["PYTHONPATH"] = (
            kinematics_src
            if not existing_pythonpath
            else (
                kinematics_src
                + os.pathsep
                + existing_pythonpath
            )
        )

        environment["PYTHONUNBUFFERED"] = "1"

        try:
            completed = subprocess.run(
                command,
                cwd=KINEMATICS_DIR,
                env=environment,
                text=True,
                capture_output=True,
                timeout=KINEMATICS_TIMEOUT_SECONDS,
                check=False,
            )

        except subprocess.TimeoutExpired as error:
            raise KinematicsBridgeError(
                "Kinematics hardware run timed out"
            ) from error

        stdout = completed.stdout.strip()
        stderr = completed.stderr.strip()

        if completed.returncode != 0:
            details = stderr or stdout or (
                "No error details were returned"
            )

            raise KinematicsBridgeError(
                "Kinematics hardware run failed "
                f"with exit code "
                f"{completed.returncode}: "
                f"{details[-3000:]}"
            )

        return {
            "accepted": True,
            "mode": "hardware_run",
            "return_code": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }

    finally:
        _hardware_lock.release()