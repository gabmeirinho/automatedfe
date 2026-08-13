"""Compatibility wrapper for the package-owned encode command."""

from automatedfe.cli.encode import main


if __name__ == "__main__":
    raise SystemExit(main())
