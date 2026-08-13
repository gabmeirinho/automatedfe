"""Backward-compatible wrapper for the transaction sorting CLI."""

from automatedfe.data.sorting import (
    DEFAULT_CARD_TOKENS_INPUT,
    DEFAULT_INPUT,
    DEFAULT_MERCHANTS_INPUT,
    DEFAULT_OUTPUT,
    DUCKDB_MEMORY_LIMIT,
    PROJECT_ROOT,
    sort_transactions,
)
from scripts.sort_transactions import main

__all__ = [
    "DEFAULT_INPUT",
    "DEFAULT_CARD_TOKENS_INPUT",
    "DEFAULT_MERCHANTS_INPUT",
    "DEFAULT_OUTPUT",
    "DUCKDB_MEMORY_LIMIT",
    "PROJECT_ROOT",
    "sort_transactions",
]


if __name__ == "__main__":
    main()
