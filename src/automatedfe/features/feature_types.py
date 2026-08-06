"""Immutable feature descriptors consumed by the complete grammar."""

from __future__ import annotations

from dataclasses import dataclass

from .feature_schema import CATEGORY_KINDS, TOTAL_TIME_WINDOW


@dataclass(frozen=True, slots=True)
class TxFeature:
    """A transaction-derived feature descriptor."""

    kind: str
    input_col: str
    window: int
    window_type: str
    category_code: int | None = None
    category_family: str | None = None

    @property
    def name(self) -> str:
        window = (
            "total"
            if self.window_type == "time" and self.window == TOTAL_TIME_WINDOW
            else str(self.window)
        )

        if self.kind in CATEGORY_KINDS:
            if self.category_family is None or self.category_code is None:
                raise ValueError(
                    "Categorical transaction features require a category family "
                    "and category code"
                )
            return (
                f"feat_{self.kind}_{self.category_family}_{self.category_code}_"
                f"{self.input_col}_{self.window_type}_{window}"
            )

        return f"feat_{self.kind}_{self.input_col}_{self.window_type}_{window}"


__all__ = ["TxFeature"]
