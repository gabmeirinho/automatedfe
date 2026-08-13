"""Compatibility adapter for the package-owned search command."""

from __future__ import annotations

from collections.abc import Sequence

from automatedfe.cli import search as _search
from automatedfe.cli.search import (
    DEFAULT_DATASET,
    DEFAULT_MAPPING,
    DEFAULT_MMAP_DIR,
    build_parser,
    run_feature_search,
)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the package command while preserving legacy monkeypatch points."""

    original = _search.run_feature_search
    _search.run_feature_search = run_feature_search
    try:
        return _search.main(argv)
    finally:
        _search.run_feature_search = original


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_MAPPING",
    "DEFAULT_MMAP_DIR",
    "build_parser",
    "main",
    "run_feature_search",
]
