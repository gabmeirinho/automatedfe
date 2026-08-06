"""Search-space catalogs used by the complete feature grammar.

The catalogs mirror the feature grammar from ``gp-benchmarks``.  They are
kept separate from the materialization kernels so that grammar construction
does not require a dataset to be present.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

CATEGORY_KINDS: Final[set[str]] = {
    "count_category",
    "avg_daily_count_category",
    "avg_daily_amount_category",
}

DAILY_KINDS: Final[set[str]] = {
    "avg_daily_count",
    "avg_daily_count_category",
    "avg_daily_amount",
    "avg_daily_amount_category",
    "avg_daily_total_amount",
}

TOTAL_TIME_WINDOW: Final[int] = -1
DAY_MICROSECONDS: Final[int] = 24 * 60 * 60 * 1_000_000

TX_ROW_WINDOWS: Final[list[int]] = [5, 10, 20, 50, 100, 200, 400, 800, 1600]
TX_TIME_WINDOWS: Final[list[int]] = [
    3600 * 1_000_000,
    6 * 3600 * 1_000_000,
    24 * 3600 * 1_000_000,
    7 * 24 * 3600 * 1_000_000,
    14 * 24 * 3600 * 1_000_000,
    30 * 24 * 3600 * 1_000_000,
    60 * 24 * 3600 * 1_000_000,
    90 * 24 * 3600 * 1_000_000,
    TOTAL_TIME_WINDOW,
]
TX_DAYS_WINDOWS: Final[list[int]] = [7, 14, 30]

FAMILIES: Final[list[str]] = [
    "status",
    "capture_method",
    "payment_method",
    "card_brand",
    "document_type",
]

TX_WINDOWS: Final[tuple[tuple[str, int], ...]] = tuple(
    ("row", window) for window in TX_ROW_WINDOWS
) + tuple(("time", window) for window in TX_TIME_WINDOWS)


def code_lists_from_mapping(
    mapping: Mapping[str, Mapping[str, int]],
) -> list[list[int]]:
    """Return the available encoded values for each categorical family.

    The persisted label mapping is the source of truth for category codes;
    this helper only adapts its JSON shape to the grammar's family-indexed
    representation.
    """

    missing = [family for family in FAMILIES if family not in mapping]
    if missing:
        raise ValueError(
            "Label mapping is missing categorical family/families: "
            + ", ".join(missing)
        )
    return [sorted(mapping[family].values()) for family in FAMILIES]


__all__ = [
    "CATEGORY_KINDS",
    "code_lists_from_mapping",
    "DAILY_KINDS",
    "DAY_MICROSECONDS",
    "FAMILIES",
    "TOTAL_TIME_WINDOW",
    "TX_DAYS_WINDOWS",
    "TX_ROW_WINDOWS",
    "TX_TIME_WINDOWS",
    "TX_WINDOWS",
]
