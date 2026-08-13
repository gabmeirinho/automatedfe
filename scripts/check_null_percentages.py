"""Compatibility wrapper for the package-owned null-percentage check."""

from automatedfe.cli.check_null_percentages import *  # noqa: F401,F403
from automatedfe.cli.check_null_percentages import main


if __name__ == "__main__":
    raise SystemExit(main())
