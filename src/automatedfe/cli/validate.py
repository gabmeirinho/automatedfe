"""Dispatch validation and diagnostic checks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from importlib import import_module
import sys
from types import ModuleType


_CHECKS: dict[str, tuple[str, tuple[str, ...], str]] = {
    "merchants-code-mismatches": (
        "automatedfe.cli.check_merchants_code_mismatches",
        ("merchant-code-mismatches", "check-merchants-code-mismatches"),
        "Compare transaction merchant category codes with merchants.parquet",
    ),
    "merchants-multiple-codes": (
        "automatedfe.cli.check_merchants_multiple_codes",
        ("merchant-multiple-codes", "check-merchants-multiple-codes"),
        "Find merchants with more than one merchant category code",
    ),
    "mmap-lengths": (
        "automatedfe.cli.check_mmap_lengths",
        ("check-mmap-lengths",),
        "Check mmap row counts against transformed transactions",
    ),
    "null-percentages": (
        "automatedfe.cli.check_null_percentages",
        ("check-null-percentages",),
        "Print null counts and percentages for transformed datasets",
    ),
    "nulls-transactions": (
        "automatedfe.cli.check_nulls_transactions",
        ("transaction-nulls", "check-nulls-transactions"),
        "Compare null counts for source and transformed transactions",
    ),
    "transactions-sorted": (
        "automatedfe.cli.check_transactions_sorted",
        ("check-transactions-sorted",),
        "Check transaction ordering",
    ),
}

_ALIASES = {
    alias: name for name, (_, aliases, _) in _CHECKS.items() for alias in aliases
}


def build_parser() -> argparse.ArgumentParser:
    """Build the validation dispatcher parser."""

    parser = argparse.ArgumentParser(
        prog="automatedfe validate",
        description="Run a data validation or diagnostic check.",
    )
    subparsers = parser.add_subparsers(dest="check", required=True, title="checks")
    for name, (_, aliases, help_text) in _CHECKS.items():
        subparsers.add_parser(name, aliases=list(aliases), help=help_text)
    return parser


def _module_for(check: str) -> ModuleType:
    canonical = _ALIASES.get(check, check)
    module_name = _CHECKS[canonical][0]
    return import_module(module_name)


def main(argv: Sequence[str] | None = None) -> int:
    """Route a validation check and return its status code."""

    parser = build_parser()
    arguments = list(argv) if argv is not None else sys.argv[1:]
    if not arguments:
        parser.parse_args(arguments)
        return 2  # argparse exits for the required subcommand above.

    check = arguments[0]
    if check.startswith("-") or (check not in _CHECKS and check not in _ALIASES):
        parser.parse_args(arguments)
        return 2

    module = _module_for(check)
    return int(module.main(arguments[1:]))


__all__ = ["build_parser", "main"]
