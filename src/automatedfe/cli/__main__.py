"""Allow ``python -m automatedfe.cli`` to use the unified dispatcher."""

from .main import main


if __name__ == "__main__":
    raise SystemExit(main())
