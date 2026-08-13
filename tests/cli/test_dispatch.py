import importlib

import pytest


dispatch = importlib.import_module("automatedfe.cli.main")
validate = importlib.import_module("automatedfe.cli.validate")


def test_top_level_parser_lists_unified_commands():
    help_text = dispatch.build_parser().format_help()

    for command in (
        "preprocess",
        "sort-transactions",
        "sort-dataset",
        "encode",
        "materialize",
        "search",
        "validate",
    ):
        assert command in help_text


def test_dispatch_routes_to_the_requested_command(monkeypatch):
    command_module = importlib.import_module("automatedfe.cli.sort_dataset")
    calls = []

    def fake_main(argv):
        calls.append(argv)
        return 7

    monkeypatch.setattr(command_module, "main", fake_main)

    assert dispatch.main(["sort-dataset", "--sentinel"]) == 7
    assert calls == [["--sentinel"]]


def test_dispatch_routes_validation_checks(monkeypatch):
    check_module = importlib.import_module(
        "automatedfe.cli.check_transactions_sorted"
    )
    calls = []

    def fake_main(argv):
        calls.append(argv)
        return 3

    monkeypatch.setattr(check_module, "main", fake_main)

    assert validate.main(["transactions-sorted", "--input", "input.parquet"]) == 3
    assert calls == [["--input", "input.parquet"]]


@pytest.mark.parametrize("arguments", [[], ["not-a-command"]])
def test_dispatch_rejects_missing_or_unknown_top_level_commands(arguments):
    with pytest.raises(SystemExit) as error:
        dispatch.main(arguments)

    assert error.value.code == 2


def test_validation_dispatch_rejects_unknown_checks():
    with pytest.raises(SystemExit) as error:
        validate.main(["not-a-check"])

    assert error.value.code == 2
