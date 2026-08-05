"""Grammar used by the feature-search genetic program."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from geneticengine.grammar import extract_grammar
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


@dataclass
class Mean:
    """Mean aggregation with exactly two parameters: feature and window."""

    feature: Feature
    window: WindowIndex

    @property
    def selected_window(self) -> FeatureWindow:
        return WINDOW_CATALOG[self.window]

    def to_feature_spec(self) -> FeatureSpec:
        return FeatureSpec(Aggregation.MEAN, self.feature, self.selected_window)

    def __str__(self) -> str:
        return self.to_feature_spec().name


def build_grammar() -> Grammar:
    """Create the depth-1 ``Mean(feature, window)`` grammar."""

    return extract_grammar([Mean], Mean)


__all__ = [
    "Feature",
    "Mean",
    "WindowIndex",
    "build_grammar",
]
