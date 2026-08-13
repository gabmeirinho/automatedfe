"""Canonical model-evaluation APIs.

Evaluation implementations are still located in their pre-migration modules
until the evaluation move in phase 2.  This package exposes the intended
evaluation boundary without duplicating those implementations.
"""

from ..features.final_evaluation import (
    ARCHIVE_MINIMIZE,
    AdditiveEvaluationResult,
    ArchiveSource,
    FinalEvaluationResult,
    FinalEvaluator,
    TEST_SPLIT,
)
from ..features.fitness import (
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
