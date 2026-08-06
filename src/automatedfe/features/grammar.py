"""Complete feature-search grammar.

This module contains the full expression grammar used by ``gp-benchmarks``:

* transaction amount and count aggregates;
* category-specific and daily transaction aggregates;
* category rates;
* arithmetic expression nodes with safe division and signed logarithms.

The complete grammar is intentionally independent of feature materialization.
Its leaves resolve to immutable :class:`TxFeature` descriptors, which can
later be materialized by the appropriate backend.
The smaller :func:`build_transaction_grammar` remains available for the
currently implemented sliding-window kernels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Annotated, ClassVar

import numpy as np
from geneticengine.grammar import extract_grammar
from geneticengine.grammar.decorators import abstract
from geneticengine.grammar.metahandlers.dependent import Dependent
from geneticengine.grammar.metahandlers.ints import IntRange
from geneticengine.grammar.metahandlers.vars import VarRange

from .feature_schema import (
    FAMILIES,
    TX_DAYS_WINDOWS,
    TX_ROW_WINDOWS,
    TX_TIME_WINDOWS,
    WINDOW_TYPE,
    code_lists_from_mapping,
)
from .feature_spec import (
    AMOUNT_COLUMN,
    WINDOW_CATALOG,
    Aggregation,
    FeatureSpec,
)
from .feature_types import TxFeature


_code_lists: tuple[tuple[int, ...], ...] | None = None


def _load_label_mapping(
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None,
) -> Mapping[str, Mapping[str, int]]:
    if mapping is None:
        from ..encoding import DEFAULT_MAPPING_OUTPUT, load_label_mapping

        return load_label_mapping(DEFAULT_MAPPING_OUTPUT)
    if isinstance(mapping, (str, PathLike)):
        from ..encoding import load_label_mapping

        return load_label_mapping(Path(mapping))
    return mapping


def _configure_code_lists(
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str],
) -> None:
    global _code_lists
    _code_lists = tuple(
        tuple(values) for values in code_lists_from_mapping(_load_label_mapping(mapping))
    )


def _get_code_lists() -> tuple[tuple[int, ...], ...]:
    if _code_lists is None:
        _configure_code_lists(None)
    assert _code_lists is not None
    return _code_lists


class expr(ABC):
    """Base type for expression nodes."""


@abstract
class ArithmeticOp(expr, ABC):
    """Abstract arithmetic expression node."""


@dataclass
class Add(ArithmeticOp):
    left: expr
    right: expr

    def evaluate(self, feature_values):
        return self.left.evaluate(feature_values) + self.right.evaluate(feature_values)

    def __str__(self) -> str:
        return f"({self.left} + {self.right})"


@dataclass
class Sub(ArithmeticOp):
    left: expr
    right: expr

    def evaluate(self, feature_values):
        return self.left.evaluate(feature_values) - self.right.evaluate(feature_values)

    def __str__(self) -> str:
        return f"({self.left} - {self.right})"


@dataclass
class Log(ArithmeticOp):
    value: expr

    def evaluate(self, feature_values):
        value = self.value.evaluate(feature_values)
        return np.sign(value) * np.log1p(np.abs(value))

    def __str__(self) -> str:
        return f"signed_log({self.value})"


@dataclass
class Mul(ArithmeticOp):
    left: expr
    right: expr

    def evaluate(self, feature_values):
        return np.clip(
            self.left.evaluate(feature_values) * self.right.evaluate(feature_values),
            -1e6,
            1e6,
        )

    def __str__(self) -> str:
        return f"({self.left} * {self.right})"


@dataclass
class SafeDiv(ArithmeticOp):
    left: expr
    right: expr

    def evaluate(self, feature_values):
        numerator = self.left.evaluate(feature_values)
        denominator = self.right.evaluate(feature_values)
        result = np.ones_like(numerator, dtype=np.float32)
        np.divide(
            numerator,
            denominator,
            out=result,
            where=np.abs(denominator) >= 1e-6,
        )
        return result

    def __str__(self) -> str:
        return f"({self.left} / {self.right})"


@abstract
@dataclass
class Agg(expr, ABC):
    """Abstract feature leaf."""

    @abstractmethod
    def to_feature_spec(self) -> TxFeature:
        raise NotImplementedError

    def evaluate(self, feature_values):
        feature = self.to_feature_spec()
        return feature_values[feature.name]

    def __str__(self) -> str:
        return self.to_feature_spec().name


@abstract
class AmountAgg(Agg, ABC):
    """Abstract transaction amount aggregate."""


@abstract
class DailyAgg(Agg, ABC):
    """Abstract daily aggregate."""


@abstract
class RateAgg(Agg, ABC):
    """Abstract rate feature."""


@abstract
class CountAgg(Agg, ABC):
    """Abstract transaction count aggregate."""


def _resolve_window(window_type_i: int, row_window_i: int, time_window_i: int):
    window_type = WINDOW_TYPE[window_type_i]
    window = (
        TX_ROW_WINDOWS[row_window_i]
        if window_type == "row"
        else TX_TIME_WINDOWS[time_window_i]
    )
    return window_type, window


def _resolve_category(category_family_i: int, category_code_i: int):
    code_lists = _get_code_lists()
    family = FAMILIES[category_family_i]
    code = code_lists[category_family_i][category_code_i]
    return family, code


def _category_code_range(category_family_i: int):
    return IntRange(0, len(_get_code_lists()[category_family_i]) - 1)


@dataclass
class MeanAmount(AmountAgg):
    window_type_i: Annotated[int, IntRange(0, len(WINDOW_TYPE) - 1)]
    row_window_i: Annotated[int, IntRange(0, len(TX_ROW_WINDOWS) - 1)]
    time_window_i: Annotated[int, IntRange(0, len(TX_TIME_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        window_type, window = _resolve_window(
            self.window_type_i, self.row_window_i, self.time_window_i
        )
        return TxFeature(kind="mean", input_col="amount", window=window, window_type=window_type)


@dataclass
class MaxAmount(AmountAgg):
    window_type_i: Annotated[int, IntRange(0, len(WINDOW_TYPE) - 1)]
    row_window_i: Annotated[int, IntRange(0, len(TX_ROW_WINDOWS) - 1)]
    time_window_i: Annotated[int, IntRange(0, len(TX_TIME_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        window_type, window = _resolve_window(
            self.window_type_i, self.row_window_i, self.time_window_i
        )
        return TxFeature(kind="max", input_col="amount", window=window, window_type=window_type)


@dataclass
class CountTotal(CountAgg):
    time_window_i: Annotated[int, IntRange(0, len(TX_TIME_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        return TxFeature(
            kind="count_total",
            input_col="amount",
            window=TX_TIME_WINDOWS[self.time_window_i],
            window_type="time",
        )


@dataclass
class CountCategory(CountAgg):
    category_family_i: Annotated[int, IntRange(0, len(FAMILIES) - 1)]
    category_code_i: Annotated[int, Dependent("category_family_i", _category_code_range)]
    window_type_i: Annotated[int, IntRange(0, len(WINDOW_TYPE) - 1)]
    row_window_i: Annotated[int, IntRange(0, len(TX_ROW_WINDOWS) - 1)]
    time_window_i: Annotated[int, IntRange(0, len(TX_TIME_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        window_type, window = _resolve_window(
            self.window_type_i, self.row_window_i, self.time_window_i
        )
        family_i = self.category_family_i
        code_lists = _get_code_lists()
        return TxFeature(
            kind="count_category",
            input_col="amount",
            window=window,
            window_type=window_type,
            category_family=FAMILIES[family_i],
            category_code=code_lists[family_i][self.category_code_i],
        )


@dataclass
class AvgDailyCount(DailyAgg):
    days_window_i: Annotated[int, IntRange(0, len(TX_DAYS_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        return TxFeature(
            kind="avg_daily_count",
            input_col="amount",
            window=TX_DAYS_WINDOWS[self.days_window_i],
            window_type="days",
        )


@dataclass
class AvgDailyCountCategory(DailyAgg):
    category_family_i: Annotated[int, IntRange(0, len(FAMILIES) - 1)]
    category_code_i: Annotated[int, Dependent("category_family_i", _category_code_range)]
    days_window_i: Annotated[int, IntRange(0, len(TX_DAYS_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        family, code = _resolve_category(self.category_family_i, self.category_code_i)
        return TxFeature(
            kind="avg_daily_count_category",
            input_col="amount",
            window=TX_DAYS_WINDOWS[self.days_window_i],
            window_type="days",
            category_family=family,
            category_code=code,
        )


@dataclass
class TotalAmount(AmountAgg):
    window_type_i: Annotated[int, IntRange(0, len(WINDOW_TYPE) - 1)]
    row_window_i: Annotated[int, IntRange(0, len(TX_ROW_WINDOWS) - 1)]
    time_window_i: Annotated[int, IntRange(0, len(TX_TIME_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        window_type, window = _resolve_window(
            self.window_type_i, self.row_window_i, self.time_window_i
        )
        return TxFeature(kind="total_amount", input_col="amount", window=window, window_type=window_type)


@dataclass
class StdAmount(AmountAgg):
    window_type_i: Annotated[int, IntRange(0, len(WINDOW_TYPE) - 1)]
    row_window_i: Annotated[int, IntRange(0, len(TX_ROW_WINDOWS) - 1)]
    time_window_i: Annotated[int, IntRange(0, len(TX_TIME_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        window_type, window = _resolve_window(
            self.window_type_i, self.row_window_i, self.time_window_i
        )
        return TxFeature(kind="std", input_col="amount", window=window, window_type=window_type)


@dataclass
class AvgDailyAmount(DailyAgg):
    days_window_i: Annotated[int, IntRange(0, len(TX_DAYS_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        return TxFeature(
            kind="avg_daily_amount",
            input_col="amount",
            window=TX_DAYS_WINDOWS[self.days_window_i],
            window_type="days",
        )


@dataclass
class AvgDailyAmountCategory(DailyAgg):
    category_family_i: Annotated[int, IntRange(0, len(FAMILIES) - 1)]
    category_code_i: Annotated[int, Dependent("category_family_i", _category_code_range)]
    days_window_i: Annotated[int, IntRange(0, len(TX_DAYS_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        family, code = _resolve_category(self.category_family_i, self.category_code_i)
        return TxFeature(
            kind="avg_daily_amount_category",
            input_col="amount",
            window=TX_DAYS_WINDOWS[self.days_window_i],
            window_type="days",
            category_family=family,
            category_code=code,
        )


@dataclass
class AvgDailyTotalAmount(DailyAgg):
    days_window_i: Annotated[int, IntRange(0, len(TX_DAYS_WINDOWS) - 1)]

    def to_feature_spec(self) -> TxFeature:
        return TxFeature(
            kind="avg_daily_total_amount",
            input_col="amount",
            window=TX_DAYS_WINDOWS[self.days_window_i],
            window_type="days",
        )


@dataclass
class CategoryRate(RateAgg):
    category_family_i: Annotated[int, IntRange(0, len(FAMILIES) - 1)]
    category_code_i: Annotated[int, Dependent("category_family_i", _category_code_range)]
    time_window_i: Annotated[int, IntRange(0, len(TX_TIME_WINDOWS) - 1)]

    def _count_category_feature(self) -> TxFeature:
        family_i = self.category_family_i
        code_lists = _get_code_lists()
        return TxFeature(
            kind="count_category",
            input_col="amount",
            window=TX_TIME_WINDOWS[self.time_window_i],
            window_type="time",
            category_family=FAMILIES[family_i],
            category_code=code_lists[family_i][self.category_code_i],
        )

    def _count_total_feature(self) -> TxFeature:
        return TxFeature(
            kind="count_total",
            input_col="amount",
            window=TX_TIME_WINDOWS[self.time_window_i],
            window_type="time",
        )

    def to_feature_spec(self) -> TxFeature:
        return self._count_category_feature()

    def evaluate(self, feature_values):
        numerator = feature_values[self._count_category_feature().name]
        denominator = feature_values[self._count_total_feature().name]
        result = np.zeros_like(numerator, dtype=np.float32)
        np.divide(numerator, denominator, out=result, where=np.abs(denominator) >= 1e-6)
        return result


TERMINALS = (
    MeanAmount,
    MaxAmount,
    CountTotal,
    CountCategory,
    TotalAmount,
    AvgDailyCount,
    AvgDailyCountCategory,
    AvgDailyAmount,
    AvgDailyAmountCategory,
    AvgDailyTotalAmount,
    CategoryRate,
    StdAmount,
)

NON_TERMINALS = (Add, Sub, Mul, SafeDiv, Log)


def collect_features(node: expr) -> set[TxFeature]:
    """Collect the primitive materialization dependencies of an expression."""

    if isinstance(node, CategoryRate):
        return {node._count_category_feature(), node._count_total_feature()}
    if isinstance(node, Log):
        return collect_features(node.value)
    if isinstance(node, TERMINALS):
        return {node.to_feature_spec()}
    if isinstance(node, NON_TERMINALS):
        return collect_features(node.left) | collect_features(node.right)
    return set()


def build_grammar(
    mapping: Mapping[str, Mapping[str, int]] | str | PathLike[str] | None = None,
):
    """Build the complete expression grammar from the persisted label mapping.

    If *mapping* is omitted, the default mapping produced by the preprocessing
    pipeline is loaded. Passing a mapping or JSON path reconfigures the
    categorical search space for that dataset.
    """

    if mapping is not None or _code_lists is None:
        _configure_code_lists(mapping)

    return extract_grammar([*TERMINALS, *NON_TERMINALS], expr)


# The original refactor's first GP implementation only supports the four
# sliding-window aggregation nodes. Keep that grammar available while the
# additional transaction-category materializers are ported.
Feature = Annotated[str, VarRange([AMOUNT_COLUMN])]
WindowIndex = Annotated[int, IntRange(0, len(WINDOW_CATALOG) - 1)]


@abstract
class AggregationFeature:
    """Abstract root for the currently supported basic transaction grammar."""

    aggregation: ClassVar[Aggregation]

    @property
    def selected_window(self):
        return WINDOW_CATALOG[self.window]

    def to_feature_spec(self) -> FeatureSpec:
        input_column = None if self.aggregation is Aggregation.COUNT else self.feature
        return FeatureSpec(self.aggregation, input_column, self.selected_window)

    def __str__(self) -> str:
        return self.to_feature_spec().name


@dataclass
class Count(AggregationFeature):
    aggregation = Aggregation.COUNT
    window: WindowIndex

    @property
    def feature(self) -> None:
        return None


@dataclass
class Sum(AggregationFeature):
    aggregation = Aggregation.SUM
    feature: Feature
    window: WindowIndex


@dataclass
class Mean(AggregationFeature):
    aggregation = Aggregation.MEAN
    feature: Feature
    window: WindowIndex


@dataclass
class Max(AggregationFeature):
    aggregation = Aggregation.MAX
    feature: Feature
    window: WindowIndex


def build_transaction_grammar():
    """Build the four-node grammar supported by the current kernels."""

    return extract_grammar([Count, Sum, Mean, Max], AggregationFeature)


def count_nodes(node: expr) -> int:
    """Return the number of nodes in a complete-grammar expression."""

    if isinstance(node, Log):
        return 1 + count_nodes(node.value)
    if isinstance(node, TERMINALS):
        return 1
    if isinstance(node, NON_TERMINALS):
        return 1 + count_nodes(node.left) + count_nodes(node.right)
    raise TypeError(f"Unknown node type: {type(node)}")


def tree_depth(node: expr) -> int:
    """Return the depth of a complete-grammar expression."""

    if isinstance(node, Log):
        return 1 + tree_depth(node.value)
    if isinstance(node, TERMINALS):
        return 1
    if isinstance(node, NON_TERMINALS):
        return 1 + max(tree_depth(node.left), tree_depth(node.right))
    raise TypeError(f"Unknown node type: {type(node)}")


__all__ = [
    "Add",
    "Agg",
    "AggregationFeature",
    "AmountAgg",
    "ArithmeticOp",
    "AvgDailyAmount",
    "AvgDailyAmountCategory",
    "AvgDailyCount",
    "AvgDailyCountCategory",
    "AvgDailyTotalAmount",
    "CategoryRate",
    "Count",
    "CountAgg",
    "CountCategory",
    "CountTotal",
    "DailyAgg",
    "Feature",
    "Log",
    "Max",
    "MaxAmount",
    "MeanAmount",
    "Mul",
    "NON_TERMINALS",
    "RateAgg",
    "SafeDiv",
    "StdAmount",
    "Sub",
    "TERMINALS",
    "TotalAmount",
    "WindowIndex",
    "build_grammar",
    "build_transaction_grammar",
    "collect_features",
    "code_lists_from_mapping",
    "count_nodes",
    "expr",
    "tree_depth",
]
