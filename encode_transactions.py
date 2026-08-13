"""Backward-compatible wrapper for the transaction label-encoding CLI."""

from automatedfe.data.encoding import (
    DEFAULT_DATASET_INPUT,
    DEFAULT_INPUT,
    DEFAULT_MAPPING_OUTPUT,
    DEFAULT_OUTPUT,
    encode_transactions,
    fit_label_mapping,
    load_label_mapping,
)
from scripts.encode_transactions import main

__all__ = [
    "DEFAULT_DATASET_INPUT",
    "DEFAULT_INPUT",
    "DEFAULT_MAPPING_OUTPUT",
    "DEFAULT_OUTPUT",
    "encode_transactions",
    "fit_label_mapping",
    "load_label_mapping",
]


if __name__ == "__main__":
    main()
