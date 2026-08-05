"""Grammar used by the feature-search genetic program.

The grammar exposes exactly the aggregations implemented by the sliding-window
kernels: count, sum, mean, and max. All value aggregations use the amount
column; count deliberately has no input column because the count kernel only
needs merchant ids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, ClassVar

from geneticengine.grammar import extract_grammar
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.grammar import Grammar
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.vars import VarRange

from .feature_spec import (
    AMOUNT_COLUMN,
    WINDOW_CATALOG,
    Aggregation,
    FeatureSpec,
)
from .feature_spec import Window as FeatureWindow

# Constrained terminal types keep Mean at tree depth 1. Adding another base
# feature later only requires adding its column name to VarRange.
Feature = Annotated[str, VarRange([AMOUNT_COLUMN])]
WindowIndex = Annotated[int, IntRange(0, len(WINDOW_CATALOG) - 1)]


@abstract
class AggregationFeature:
    """Abstract root for one aggregation over one catalogued window."""

    aggregation: ClassVar[Aggregation]

    @property
    def selected_window(self) -> FeatureWindow:
        return WINDOW_CATALOG[self.window]

    def to_feature_spec(self) -> FeatureSpec:
        input_column = None if self.aggregation is Aggregation.COUNT else self.feature
        return FeatureSpec(self.aggregation, input_column, self.selected_window)

    def __str__(self) -> str:
        return self.to_feature_spec().name


@dataclass
class Count(AggregationFeature):
    """Count preceding transactions in the selected window."""

    aggregation = Aggregation.COUNT
    window: WindowIndex

    @property
    def feature(self) -> None:
        """Count transactions has no source value column."""

        return None


@dataclass
class Sum(AggregationFeature):
    """Sum the amount column over the selected window."""

    aggregation = Aggregation.SUM
    feature: Feature
    window: WindowIndex


@dataclass
class Mean(AggregationFeature):
    """Mean aggregation with exactly two parameters: feature and window."""

    aggregation = Aggregation.MEAN
    feature: Feature
    window: WindowIndex


@dataclass
class Max(AggregationFeature):
    """Take the maximum amount over the selected window."""

    aggregation = Aggregation.MAX
    feature: Feature
    window: WindowIndex


def build_grammar() -> Grammar:
    """Create the aggregation grammar used by the feature search."""

    return extract_grammar([Count, Sum, Mean, Max], AggregationFeature)


__all__ = [
    "AggregationFeature",
    "Count",
    "Feature",
    "Max",
    "Mean",
    "Sum",
    "WindowIndex",
    "build_grammar",
]
