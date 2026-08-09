"""Reusable dataset sorting, validation, and encoding functions."""

from .encoding import (
    CATEGORICAL_COLUMNS,
    DEFAULT_MAPPING_OUTPUT,
    encode_transactions,
    fit_label_mapping,
    load_label_mapping,
)
from .features.archive import ArchiveSnapshot, ArchiveStep, load_archive
from .features.feature_materialization import (
    FeatureMaterializer,
    materialize_aggregation,
    materialize_feature,
    materialize_individual,
)
from .fitness import (
    FitnessEvaluator,
    RandomForestFitness,
    ResidualEvaluator,
    ResidualFitness,
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
    DEFAULT_MMAP_DIR,
    column_dtype,
    load_mmapped_columns,
    materialize_transactions,
    read_manifest,
)
from .validation import first_sorting_violation

__all__ = [
    "ArchiveSnapshot",
    "ArchiveStep",
    "CATEGORICAL_COLUMNS",
    "DEFAULT_CARD_TOKENS_INPUT",
    "DEFAULT_DATASET_INPUT",
    "DEFAULT_DATASET_OUTPUT",
    "DEFAULT_INPUT",
    "DEFAULT_MAPPING_OUTPUT",
    "DEFAULT_MERCHANTS_INPUT",
    "DEFAULT_MMAP_DIR",
    "DEFAULT_OUTPUT",
    "DUCKDB_MEMORY_LIMIT",
    "PROJECT_ROOT",
    "FeatureMaterializer",
    "FitnessEvaluator",
    "RandomForestFitness",
    "ResidualEvaluator",
    "ResidualFitness",
    "column_dtype",
    "encode_transactions",
    "first_sorting_violation",
    "fit_label_mapping",
    "load_label_mapping",
    "load_archive",
    "load_mmapped_columns",
    "materialize_aggregation",
    "materialize_feature",
    "materialize_individual",
    "materialize_transactions",
    "preprocess",
    "read_manifest",
    "sort_dataset",
    "sort_transactions",
]
