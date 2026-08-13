"""Compatibility wrapper for the package-owned sorting check."""

from automatedfe.cli.check_transactions_sorted import *  # noqa: F401,F403
from automatedfe.cli.check_transactions_sorted import main


if __name__ == "__main__":
    raise SystemExit(main())
