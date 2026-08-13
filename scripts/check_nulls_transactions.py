"""Compatibility wrapper for the package-owned transaction-null check."""

from automatedfe.cli.check_nulls_transactions import *  # noqa: F401,F403
from automatedfe.cli.check_nulls_transactions import main


if __name__ == "__main__":
    raise SystemExit(main())
