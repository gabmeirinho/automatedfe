import numpy as np
import pytest

from automatedfe.features.feature_schema import TX_ROW_WINDOWS
from automatedfe.features.grammar import (
    Add,
    Agg,
    ArithmeticOp,
    AvgDailyAmount,
    AvgDailyAmountCategory,
    AvgDailyCount,
    AvgDailyCountCategory,
    AvgDailyTotalAmount,
    CategoryRate,
    CountAgg,
    CountCategory,
    CountTotal,
    Log,
    MeanAmount,
    Mul,
    SafeDiv,
    StdAmount,
    Sub,
    TotalAmount,
    build_grammar,
    collect_features,
    count_nodes,
    expr,
    tree_depth,
)


LABEL_MAPPING = {
    "status": {"approved": 0, "complete": 1, "denied": 2, "others": 3},
    "capture_method": {"contactless": 0, "emv": 1, "pix": 2},
    "payment_method": {"debit": 0, "credit": 1, "null": -1},
    "card_brand": {"mastercard": 0, "visa": 1, "null": -1},
    "document_type": {"cnpj": 0, "cpf": 1, "null": -1},
}


@pytest.fixture(autouse=True)
def configure_label_mapping():
    build_grammar(LABEL_MAPPING)


def test_complete_grammar_contains_transaction_and_arithmetic_branches():
    grammar = build_grammar(LABEL_MAPPING)

    assert grammar.starting_symbol is expr
    assert set(grammar.alternatives[expr]) == {Agg, ArithmeticOp}

    assert {
        CountTotal,
        CountCategory,
    }.issubset(set(grammar.alternatives[CountAgg]))


def test_transaction_terminals_resolve_to_encoded_feature_names():
    assert str(MeanAmount(len(TX_ROW_WINDOWS) + 2)) == "feat_mean_amount_time_1d"
    assert str(TotalAmount(len(TX_ROW_WINDOWS) + 8)) == (
        "feat_total_amount_amount_time_total"
    )
    assert str(StdAmount(len(TX_ROW_WINDOWS) + 1)) == "feat_std_amount_time_6h"
    assert str(CountTotal(8)) == "feat_count_total_amount_time_total"
    assert str(CountCategory(0, 0, 0)) == "feat_count_category_status_0_amount_row_5"
    assert str(AvgDailyCount(0)) == "feat_avg_daily_count_amount_days_7"
    assert str(AvgDailyCountCategory(0, 0, 1)) == (
        "feat_avg_daily_count_category_status_0_amount_days_14"
    )
    assert str(AvgDailyAmount(2)) == "feat_avg_daily_amount_amount_days_30"
    assert str(AvgDailyAmountCategory(0, 0, 0)) == (
        "feat_avg_daily_amount_category_status_0_amount_days_7"
    )
    assert str(AvgDailyTotalAmount(1)) == "feat_avg_daily_total_amount_amount_days_14"


def test_category_rate_collects_both_primitive_dependencies_and_is_safe():
    rate = CategoryRate(0, 0, 1)
    dependencies = collect_features(rate)
    names = {feature.name for feature in dependencies}

    assert names == {
        "feat_count_category_status_0_amount_time_6h",
        "feat_count_total_amount_time_6h",
    }

    values = {name: np.array([2.0, 0.0, 1.0]) for name in names}
    values["feat_count_total_amount_time_6h"] = np.array([4.0, 0.0, 0.0])
    np.testing.assert_allclose(rate.evaluate(values), [0.5, 0.0, 0.0])


def test_arithmetic_nodes_evaluate_and_have_stable_string_forms():
    left = MeanAmount(0)
    right = CountTotal(0)
    values = {
        left.to_feature_spec().name: np.array([2.0, -4.0, 0.0]),
        right.to_feature_spec().name: np.array([2.0, 0.0, 0.0]),
    }

    np.testing.assert_allclose(Add(left, right).evaluate(values), [4.0, -4.0, 0.0])
    np.testing.assert_allclose(Sub(left, right).evaluate(values), [0.0, -4.0, 0.0])
    np.testing.assert_allclose(Mul(left, right).evaluate(values), [4.0, 0.0, 0.0])
    np.testing.assert_allclose(SafeDiv(left, right).evaluate(values), [1.0, 1.0, 1.0])
    np.testing.assert_allclose(Log(left).evaluate(values), np.sign([2.0, -4.0, 0.0]) * np.log1p([2.0, 4.0, 0.0]))
    assert str(Add(left, right)) == f"({left} + {right})"


def test_collect_features_handles_nested_arithmetic():
    expression = Log(Add(MeanAmount(0), TotalAmount(0)))
    dependencies = collect_features(expression)

    assert len(dependencies) == 2
    assert count_nodes(expression) == 4
    assert tree_depth(expression) == 3
