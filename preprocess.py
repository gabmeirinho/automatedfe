"""Backward-compatible wrapper for the full preprocessing CLI."""

from automatedfe.data.preprocessing import (
    DEFAULT_CARD_TOKENS_INPUT,
    DEFAULT_DATASET_INPUT,
    DEFAULT_DATASET_OUTPUT,
    DEFAULT_INPUT,
    DEFAULT_MAPPING_OUTPUT,
    DEFAULT_MERCHANTS_INPUT,
    DEFAULT_OUTPUT,
    preprocess,
)
from scripts.preprocess import main

__all__ = [
    "DEFAULT_CARD_TOKENS_INPUT",
    "DEFAULT_DATASET_INPUT",
    "DEFAULT_DATASET_OUTPUT",
    "DEFAULT_INPUT",
    "DEFAULT_MAPPING_OUTPUT",
    "DEFAULT_MERCHANTS_INPUT",
    "DEFAULT_OUTPUT",
    "preprocess",
]


if __name__ == "__main__":
    main()
