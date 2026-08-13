"""Compatibility wrapper for the package-owned preprocess command."""

from automatedfe.cli.preprocess import main


if __name__ == "__main__":
    raise SystemExit(main())
