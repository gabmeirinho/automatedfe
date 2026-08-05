"""Backward-compatible wrapper for the transaction sorting CLI."""

from automatedfe.sorting import (
    DEFAULT_INPUT,
    DEFAULT_OUTPUT,
    DUCKDB_MEMORY_LIMIT,
    PROJECT_ROOT,
    sort_transactions,
)
from scripts.sort_transactions import main

__all__ = [
    "DEFAULT_INPUT",
    "DEFAULT_OUTPUT",
    "DUCKDB_MEMORY_LIMIT",
    "PROJECT_ROOT",
    "sort_transactions",
]


if __name__ == "__main__":
    main()
