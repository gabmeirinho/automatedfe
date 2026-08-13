"""Compatibility wrapper for the package-owned sort-dataset command."""

from automatedfe.cli.sort_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
