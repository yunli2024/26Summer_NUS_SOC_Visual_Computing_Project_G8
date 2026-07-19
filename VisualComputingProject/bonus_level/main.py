"""Bonus Level entry point.

Run this file to open the pose detection and dance scoring GUI.
"""

try:
    from .src.app import run_app
except ImportError:
    from src.app import run_app


def main() -> int:
    return run_app()


if __name__ == "__main__":
    raise SystemExit(main())
