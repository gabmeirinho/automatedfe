"""Compatibility wrapper for the package-owned sort-transactions command."""

from automatedfe.cli.sort_transactions import main


if __name__ == "__main__":
    raise SystemExit(main())
