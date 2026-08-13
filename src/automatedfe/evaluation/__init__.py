"""Canonical model-evaluation APIs.

Fitness and held-out evaluation implementations live in this package. Feature
construction remains under :mod:`automatedfe.features`; search orchestration
consumes these evaluators through this canonical boundary.
"""

from .final_evaluation import (
    ARCHIVE_MINIMIZE,
    AdditiveEvaluationResult,
    ArchiveSource,
    FinalEvaluationResult,
    FinalEvaluator,
    TEST_SPLIT,
)
from .fitness import (
    DATASET_MERCHANT_COLUMN,
    DATASET_SPLIT_COLUMN,
    DATASET_TARGET_COLUMN,
    DATASET_TIMESTAMP_COLUMN,
    DEFAULT_N_ESTIMATORS,
    DEFAULT_N_SPLITS,
    DEFAULT_RANDOM_STATE,
    MIN_LOGIT_WEIGHT,
    RESIDUAL_EPSILON,
    RESIDUAL_SHRINKAGE,
    RESIDUAL_TREE_PARAMS,
    TRAIN_SPLIT,
    ActiveResidualEvaluator,
    ActiveResidualFitness,
    FitnessEvaluator,
    NumericalFitnessError,
    RandomForestFitness,
    ResidualEvaluator,
    ResidualFitness,
    logit,
    logit_working_response,
    objectives_are_finite,
    sigmoid,
)

__all__ = [
    "ARCHIVE_MINIMIZE",
    "DATASET_MERCHANT_COLUMN",
    "DATASET_SPLIT_COLUMN",
    "DATASET_TARGET_COLUMN",
    "DATASET_TIMESTAMP_COLUMN",
    "DEFAULT_N_ESTIMATORS",
    "DEFAULT_N_SPLITS",
    "DEFAULT_RANDOM_STATE",
    "MIN_LOGIT_WEIGHT",
    "RESIDUAL_EPSILON",
    "RESIDUAL_SHRINKAGE",
    "RESIDUAL_TREE_PARAMS",
    "TEST_SPLIT",
    "TRAIN_SPLIT",
    "ActiveResidualEvaluator",
    "ActiveResidualFitness",
    "AdditiveEvaluationResult",
    "ArchiveSource",
    "FinalEvaluationResult",
    "FinalEvaluator",
    "FitnessEvaluator",
    "NumericalFitnessError",
    "RandomForestFitness",
    "ResidualEvaluator",
    "ResidualFitness",
    "logit",
    "logit_working_response",
    "objectives_are_finite",
    "sigmoid",
]
