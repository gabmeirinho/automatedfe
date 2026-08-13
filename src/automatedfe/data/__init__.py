"""Canonical data-pipeline APIs.

This namespace owns sorting, validation, encoding, preprocessing, and
transaction materialization.  The package-level re-exports intentionally
point at these implementations so callers can use one coherent data API.
"""

from .encoding import (
    CATEGORICAL_COLUMNS,
    DEFAULT_MAPPING_OUTPUT,
    TRAIN_SPLIT,
    encode_transactions,
    fit_label_mapping,
    load_label_mapping,
)
from .preprocessing import preprocess
from .sorting import (
    DEFAULT_CARD_TOKENS_INPUT,
    DEFAULT_DATASET_INPUT,
    DEFAULT_DATASET_OUTPUT,
    DEFAULT_INPUT,
    DEFAULT_MERCHANTS_INPUT,
    DEFAULT_OUTPUT,
    DUCKDB_MEMORY_LIMIT,
    PROJECT_ROOT,
    sort_dataset,
    sort_transactions,
)
from .transaction_materialization import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MMAP_DIR,
    MANIFEST_FILENAME,
    MMAP_SUFFIX,
    column_dtype,
    load_mmapped_columns,
    materialize_transactions,
    read_manifest,
)
from .validation import first_sorting_violation

__all__ = [
    "CATEGORICAL_COLUMNS",
    "DEFAULT_CARD_TOKENS_INPUT",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_DATASET_INPUT",
    "DEFAULT_DATASET_OUTPUT",
    "DEFAULT_INPUT",
    "DEFAULT_MAPPING_OUTPUT",
    "DEFAULT_MERCHANTS_INPUT",
    "DEFAULT_MMAP_DIR",
    "DEFAULT_OUTPUT",
    "DUCKDB_MEMORY_LIMIT",
    "MANIFEST_FILENAME",
    "MMAP_SUFFIX",
    "PROJECT_ROOT",
    "TRAIN_SPLIT",
    "column_dtype",
    "encode_transactions",
    "first_sorting_violation",
    "fit_label_mapping",
    "load_label_mapping",
    "load_mmapped_columns",
    "materialize_transactions",
    "preprocess",
    "read_manifest",
    "sort_dataset",
    "sort_transactions",
]
