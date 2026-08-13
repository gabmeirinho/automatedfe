"""Compatibility wrapper for the package-owned merchant-code check."""

from automatedfe.cli.check_merchants_multiple_codes import *  # noqa: F401,F403
from automatedfe.cli.check_merchants_multiple_codes import main


if __name__ == "__main__":
    raise SystemExit(main())
