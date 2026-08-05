"""Backward-compatible wrapper for the dataset sorting CLI."""

from automatedfe.sorting import (
    DEFAULT_DATASET_INPUT as DEFAULT_INPUT,
    DEFAULT_DATASET_OUTPUT as DEFAULT_OUTPUT,
    PROJECT_ROOT,
    sort_dataset,
)
from scripts.sort_dataset import main

__all__ = ["DEFAULT_INPUT", "DEFAULT_OUTPUT", "PROJECT_ROOT", "sort_dataset"]


if __name__ == "__main__":
    main()
