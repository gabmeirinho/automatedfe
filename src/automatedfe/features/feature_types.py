"""The single immutable feature descriptor used by grammar and materializers."""

from __future__ import annotations

from dataclasses import dataclass

from .feature_schema import CATEGORY_KINDS, DAY_MICROSECONDS, TOTAL_TIME_WINDOW

_HOUR_MICROSECONDS = 3_600 * 1_000_000
_MINUTE_MICROSECONDS = 60 * 1_000_000


def _format_time_window(microseconds: int) -> str:
    for divisor, suffix in (
        (DAY_MICROSECONDS, "d"),
        (_HOUR_MICROSECONDS, "h"),
        (_MINUTE_MICROSECONDS, "m"),
    ):
        if microseconds % divisor == 0:
            return f"{microseconds // divisor}{suffix}"
    return str(microseconds)


@dataclass(frozen=True, slots=True)
class TxFeature:
    """A primitive transaction-derived feature descriptor.

    ``window`` is measured in rows for ``window_type='row'``, microseconds
    for ``'time'``, and days for ``'days'``.  A time window of ``-1`` denotes
    the complete transaction history.
    """

    kind: str
    input_col: str | None
    window: int
    window_type: str
    category_code: int | None = None
    category_family: str | None = None

    def __post_init__(self) -> None:
        if self.window_type not in {"row", "time", "days"}:
            raise ValueError(f"Unsupported window type: {self.window_type!r}")
        if self.window_type == "time" and self.window == TOTAL_TIME_WINDOW:
            return
        if self.window <= 0:
            raise ValueError(
                f"Feature windows must be positive, got {self.window} "
                f"for window type {self.window_type!r}"
            )

    @property
    def name(self) -> str:
        if self.window_type == "time":
            window = (
                "total"
                if self.window == TOTAL_TIME_WINDOW
                else _format_time_window(self.window)
            )
        else:
            window = str(self.window)

        if self.kind in CATEGORY_KINDS:
            if self.category_family is None or self.category_code is None:
                raise ValueError(
                    "Categorical transaction features require a category family "
                    "and category code"
                )
            return (
                f"feat_{self.kind}_{self.category_family}_{self.category_code}_"
                f"{self.input_col or 'amount'}_{self.window_type}_{window}"
            )

        return f"feat_{self.kind}_{self.input_col or 'amount'}_{self.window_type}_{window}"


__all__ = ["TxFeature"]
