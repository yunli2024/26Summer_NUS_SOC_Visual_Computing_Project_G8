"""3D upper-body runner entry point.

Run this file to open the Ursina pose-controlled three-lane runner prototype.
"""

try:
    from .src.runner_3d_app import run_app
except ImportError:
    from src.runner_3d_app import run_app


def main() -> int:
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
