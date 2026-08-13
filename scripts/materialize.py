"""Compatibility wrapper for the package-owned materialize command."""

from automatedfe.cli.materialize import main


if __name__ == "__main__":
    raise SystemExit(main())
