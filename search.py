"""Backward-compatible wrapper for the feature-search CLI."""

from scripts.search import (
    DEFAULT_DATASET,
    DEFAULT_MAPPING,
    DEFAULT_MMAP_DIR,
    build_parser,
    main,
)

__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_MAPPING",
    "DEFAULT_MMAP_DIR",
    "build_parser",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
