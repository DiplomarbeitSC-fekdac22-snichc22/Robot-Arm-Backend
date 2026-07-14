from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def backend_is_running() -> bool:
    checks = [
        ["pgrep", "-f", "run_backend.py"],
        ["pgrep", "-f", "uvicorn.*app.main"],
    ]

    return any(
        subprocess.run(
            command,
            capture_output=True,
            text=True,
        ).returncode == 0
        for command in checks
    )


def run_test_group(name: str, pytest_arguments: list[str]) -> int:
    print()
    print("=" * 80)
    print(f"RUNNING {name}")
    print("=" * 80)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            *pytest_arguments,
        ],
        cwd=PROJECT_ROOT,
    )

    return result.returncode


def main() -> None:
    if backend_is_running():
        print(
            "\nERROR: The backend is currently running.\n"
            "Stop it before running integration tests because "
            "the tests need exclusive access to the camera.\n"
        )
        sys.exit(1)

    unit_result = run_test_group(
        "UNIT TESTS",
        [
            "tests/unit",
            "-v",
        ],
    )

    integration_result = run_test_group(
        "RASPBERRY PI INTEGRATION TESTS",
        [
            "tests/integration",
            "-v",
            "-s",
        ],
    )

    print()
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(
        f"Unit tests:       "
        f"{'PASSED' if unit_result == 0 else 'FAILED'}"
    )
    print(
        f"Integration tests: "
        f"{'PASSED' if integration_result == 0 else 'FAILED'}"
    )

    if unit_result == 0 and integration_result == 0:
        print("\nAll backend tests passed.")
        sys.exit(0)

    print("\nAt least one test group failed.")
    sys.exit(1)


if __name__ == "__main__":
    main()