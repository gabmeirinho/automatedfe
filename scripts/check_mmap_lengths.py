"""Compatibility wrapper for the package-owned mmap-length check."""

from automatedfe.cli.check_mmap_lengths import *  # noqa: F401,F403
from automatedfe.cli.check_mmap_lengths import main


if __name__ == "__main__":
    raise SystemExit(main())
