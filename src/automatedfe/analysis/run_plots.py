"""Deterministic, table-only figures for one structured search run."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from os import PathLike
from pathlib import Path
from typing import Final

import matplotlib

# Figure rendering must also work in headless workers and test processes.
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .artifacts import (
    CANDIDATES_FILENAME,
    GENERATIONS_FILENAME,
    read_candidates_csv,
    read_generations_csv,
)
from .run_tables import SEARCH_FOLD_METRIC_KEY, read_final_evaluation_tables

FIGURES_DIRECTORY: Final[str] = "figures"
FIGURE_DPI: Final[int] = 300
FIGURE_FILENAMES: Final[tuple[str, ...]] = (
    "01_candidate_outcomes_by_generation.png",
    "02_archive_size_changes.png",
    "03_fold_score_by_generation.png",
    "04_fold_stability.png",
    "05_materialization_time.png",
    "06_score_vs_materialization_time.png",
    "07_final_archive_fold_scores.png",
    "08_forest_impurity_importance.png",
    "09_spearman_heatmap.png",
    "10_primitive_operator_presence.png",
    "11_runtime_and_evaluations.png",
)

_FOLD_COLUMNS = ("Split1", "Split2", "Split3")
_FINAL_FOLD_COLUMNS = ("search_fold_1", "search_fold_2", "search_fold_3")
_PLOT_METADATA = {"Software": "automatedfe"}


def metric_display_name(metric: str | None) -> str:
    """Return a truthful human-readable label for a persisted search metric."""

    if metric is None or not metric.strip():
        return "Search fold score"
    normalized = metric.strip().lower()
    return {
        "roc_auc": "ROC AUC",
        "brier": "Brier improvement",
        "brier_improvement": "Brier improvement",
    }.get(normalized, normalized.replace("_", " ").title())


def fold_stability(
    frame: pd.DataFrame, columns: Sequence[str] = _FOLD_COLUMNS
) -> pd.Series:
    """Return each row's population SD across its finite fold scores."""

    folds = frame.loc[:, list(columns)].apply(pd.to_numeric, errors="coerce")
    return folds.std(axis=1, ddof=0).where(folds.notna().sum(axis=1) >= 2)


def generation_iqr(
    frame: pd.DataFrame,
    value_column: str,
    *,
    generation_column: str = "Generation",
) -> pd.DataFrame:
    """Calculate count, median, and IQR for a numeric value by generation."""

    values = (
        pd.DataFrame(
            {
                "generation": pd.to_numeric(frame[generation_column], errors="coerce"),
                "value": pd.to_numeric(frame[value_column], errors="coerce"),
            }
        )
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if values.empty:
        return pd.DataFrame(columns=("generation", "count", "q1", "median", "q3"))
    grouped = values.groupby("generation", sort=True)["value"]
    result = grouped.agg(count="count", median="median").reset_index()
    result["q1"] = grouped.quantile(0.25).to_numpy()
    result["q3"] = grouped.quantile(0.75).to_numpy()
    return result.loc[:, ["generation", "count", "q1", "median", "q3"]]


def _numeric(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[column] = pd.to_numeric(
            result[column] if column in result else pd.Series(dtype=float),
            errors="coerce",
        )
    return result


def _is_evaluation_free(candidates: pd.DataFrame) -> bool:
    statuses = set(candidates.get("Status", pd.Series(dtype=str)).dropna().astype(str))
    return bool(statuses) and statuses <= {"generated"}


def _not_applicable(ax: plt.Axes, detail: str) -> None:
    ax.set_axis_off()
    ax.text(
        0.5,
        0.55,
        "Not applicable",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.5,
        0.42,
        detail,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
        wrap=True,
    )


def _finish(ax: plt.Axes, *, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, loc="left", fontsize=11, weight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.22, linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight", metadata=_PLOT_METADATA)
    plt.close(fig)


def _candidate_outcomes(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    data = candidates.dropna(subset=["Generation"])
    if data.empty:
        _not_applicable(ax, "No generation-level candidate outcomes were persisted.")
        return
    counts = pd.crosstab(data["Generation"].astype(int), data["Status"])
    order = [
        status
        for status in (
            "evaluated",
            "invalid",
            "materialization_failed",
            "duplicate",
            "generated",
        )
        if status in counts
    ]
    counts.loc[:, order].plot(kind="bar", stacked=True, ax=ax, width=0.82)
    _finish(
        ax,
        title=f"Candidate outcomes by generation (n={len(data):,})",
        xlabel="Generation",
        ylabel="Candidate count",
    )
    ax.legend(title="Outcome", frameon=False, fontsize=8)


def _archive_changes(ax: plt.Axes, generations: pd.DataFrame) -> None:
    if generations.empty:
        _not_applicable(ax, "No generation summaries were persisted.")
        return
    x = generations["Generation"]
    ax.plot(x, generations["ArchiveSize"], marker="o", label="Archive size")
    ax.bar(x, generations["Added"], alpha=0.42, label="Added")
    _finish(
        ax,
        title=f"Permanent archive membership ({len(generations):,} generations)",
        xlabel="Generation",
        ylabel="Feature count",
    )
    ax.legend(frameon=False, fontsize=8)


def _score_by_generation(
    ax: plt.Axes, candidates: pd.DataFrame, metric_label: str, evaluation_free: bool
) -> None:
    if evaluation_free:
        _not_applicable(ax, "Evaluation-free search computes no fold scores.")
        return
    data = candidates.copy()
    data["mean_fold_score"] = data.loc[:, list(_FOLD_COLUMNS)].mean(axis=1)
    data = data.dropna(subset=["Generation", "mean_fold_score"])
    if data.empty:
        _not_applicable(ax, "No finite candidate fold scores were persisted.")
        return
    summary = data.groupby("Generation", sort=True)["mean_fold_score"].agg(
        ["max", "median"]
    )
    ax.plot(summary.index, summary["max"], marker="o", label="Best")
    ax.plot(summary.index, summary["median"], marker="o", label="Median")
    _finish(
        ax,
        title=f"Candidate {metric_label.lower()} (n={len(data):,})",
        xlabel="Generation",
        ylabel=f"Mean {metric_label}",
    )
    ax.legend(frameon=False)


def _fold_stability_plot(
    ax: plt.Axes, candidates: pd.DataFrame, metric_label: str, evaluation_free: bool
) -> None:
    if evaluation_free:
        _not_applicable(ax, "Evaluation-free search computes no fold stability.")
        return
    data = candidates.copy()
    data["fold_sd"] = fold_stability(data)
    summary = generation_iqr(data, "fold_sd")
    if summary.empty:
        _not_applicable(ax, "At least two finite fold scores are required.")
        return
    x = summary["generation"].to_numpy(dtype=float)
    ax.plot(
        x, summary["median"].to_numpy(dtype=float), marker="o", label="Median fold SD"
    )
    ax.fill_between(
        x,
        summary["q1"].to_numpy(dtype=float),
        summary["q3"].to_numpy(dtype=float),
        alpha=0.24,
        label="IQR",
    )
    _finish(
        ax,
        title=f"Fold stability ({int(summary['count'].sum()):,} candidates)",
        xlabel="Generation",
        ylabel=f"SD of {metric_label}",
    )
    ax.legend(frameon=False)


def _materialization_time(ax: plt.Axes, candidates: pd.DataFrame) -> None:
    summary = generation_iqr(candidates, "MaterializationTime")
    if summary.empty:
        _not_applicable(ax, "No completed materialization durations were persisted.")
        return
    x = summary["generation"].to_numpy(dtype=float)
    ax.plot(x, summary["median"].to_numpy(dtype=float), marker="o", label="Median")
    ax.fill_between(
        x,
        summary["q1"].to_numpy(dtype=float),
        summary["q3"].to_numpy(dtype=float),
        alpha=0.24,
        label="IQR",
    )
    _finish(
        ax,
        title=f"Materialization time ({int(summary['count'].sum()):,} candidates)",
        xlabel="Generation",
        ylabel="Materialization time (seconds)",
    )
    ax.legend(frameon=False)


def _score_vs_time(
    ax: plt.Axes, candidates: pd.DataFrame, metric_label: str, evaluation_free: bool
) -> None:
    if evaluation_free:
        _not_applicable(ax, "Evaluation-free search has no score–time diagnostic.")
        return
    data = candidates.copy()
    data["mean_fold_score"] = data.loc[:, list(_FOLD_COLUMNS)].mean(axis=1)
    data = data.dropna(subset=["mean_fold_score", "MaterializationTime"])
    if data.empty:
        _not_applicable(ax, "No candidates have both fold scores and timing.")
        return
    ax.scatter(data["MaterializationTime"], data["mean_fold_score"], s=22, alpha=0.68)
    _finish(
        ax,
        title=f"Score versus materialization time (n={len(data):,})",
        xlabel="Materialization time (seconds)",
        ylabel=f"Mean {metric_label}",
    )


def _final_fold_scores(
    ax: plt.Axes,
    features: pd.DataFrame,
    metric_label: str,
    evaluation_free: bool,
    label_column: str,
) -> None:
    data = features.dropna(subset=list(_FINAL_FOLD_COLUMNS), how="all")
    if evaluation_free or data.empty:
        detail = (
            "Evaluation-free search computes no final-archive fold scores."
            if evaluation_free
            else "No final-archive fold scores were persisted."
        )
        _not_applicable(ax, detail)
        return
    positions = np.arange(len(data))
    width = 0.24
    for index, column in enumerate(_FINAL_FOLD_COLUMNS):
        ax.bar(
            positions + (index - 1) * width,
            data[column],
            width,
            label=f"Fold {index + 1}",
        )
    ax.set_xticks(positions, data[label_column], rotation=45, ha="right", fontsize=7)
    _finish(
        ax,
        title=f"Final-archive fold scores ({len(data):,} features)",
        xlabel="Final feature",
        ylabel=metric_label,
    )
    ax.legend(frameon=False, ncols=3, fontsize=8)


def _importance(ax: plt.Axes, importances: pd.DataFrame, label_column: str) -> None:
    data = importances.nlargest(20, "importance_mean").sort_values("importance_mean")
    if data.empty:
        _not_applicable(ax, "No forest impurity importances were persisted.")
        return
    ax.barh(
        data[label_column],
        data["importance_mean"],
        xerr=data["importance_std"],
        alpha=0.82,
        capsize=2,
    )
    _finish(
        ax,
        title=f"Top {len(data)} of {len(importances):,} features by forest impurity importance",
        xlabel="Mean impurity importance (error: across-tree SD)",
        ylabel="Final feature",
    )


def _heatmap(
    ax: plt.Axes,
    correlations: pd.DataFrame,
    importances: pd.DataFrame,
    label_column: str,
    row_count: int | None,
) -> None:
    top = importances.nlargest(20, "importance_mean")
    identifiers = top["feature_id"].astype(str).tolist()
    if not identifiers:
        _not_applicable(ax, "No feature correlations were persisted.")
        return
    selected = correlations[
        correlations["feature_id"].astype(str).isin(identifiers)
        & correlations["other_feature_id"].astype(str).isin(identifiers)
    ]
    matrix = selected.pivot(
        index="feature_id", columns="other_feature_id", values="spearman"
    ).reindex(index=identifiers, columns=identifiers)
    label_map = dict(zip(top["feature_id"].astype(str), top[label_column].astype(str)))
    labels = [label_map[value] for value in identifiers]
    image = ax.imshow(
        matrix.to_numpy(dtype=float),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
        interpolation="nearest",
    )
    ax.set_xticks(range(len(labels)), labels, rotation=55, ha="right", fontsize=6)
    ax.set_yticks(range(len(labels)), labels, fontsize=6)
    count_text = "unknown rows" if row_count is None else f"{row_count:,} training rows"
    ax.set_title(
        f"Top-{len(labels)} Spearman correlation ({count_text})",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    plt.colorbar(image, ax=ax, label="Spearman correlation", fraction=0.046, pad=0.04)


def _presence_labels(expressions: Sequence[str]) -> pd.Series:
    patterns = {
        "Amount aggregate": r"feat_(?:mean|max|total|std)_amount",
        "Count aggregate": r"feat_count_",
        "Daily aggregate": r"feat_avg_daily_",
        "Category rate": r"feat_category_rate",
        "Addition (+)": r"\+",
        "Subtraction (-)": r" - ",
        "Multiplication (*)": r"\*",
        "Safe division (/)": r" / ",
        "Signed log": r"signed_log\(",
    }
    return pd.Series(
        {
            name: sum(bool(re.search(pattern, value)) for value in expressions)
            for name, pattern in patterns.items()
        },
        dtype="int64",
    )


def _primitive_presence(ax: plt.Axes, features: pd.DataFrame) -> None:
    expressions = (
        features.get("feature_label", pd.Series(dtype=str))
        .dropna()
        .astype(str)
        .tolist()
    )
    if not expressions:
        _not_applicable(ax, "No final feature expressions were persisted.")
        return
    counts = _presence_labels(expressions)
    counts = counts[counts > 0].sort_values()
    if counts.empty:
        _not_applicable(
            ax, "No recognized primitive families or operators were present."
        )
        return
    ax.barh(counts.index, counts.values)
    _finish(
        ax,
        title=f"Primitive-family and operator presence ({len(expressions):,} final features)",
        xlabel="Features containing primitive/operator",
        ylabel="Primitive family or operator",
    )
    ax.set_xlim(0, max(len(expressions), int(counts.max())))


def _runtime(ax: plt.Axes, generations: pd.DataFrame) -> None:
    if generations.empty:
        _not_applicable(ax, "No generation runtime summaries were persisted.")
        return
    x = generations["Generation"]
    runtime = ax.plot(
        x,
        generations["CumulativeRuntimeSeconds"],
        marker="o",
        color="#276FBF",
        label="Cumulative runtime",
    )
    ax.set_xlabel("Generation")
    ax.set_ylabel("Cumulative runtime (seconds)", color="#276FBF")
    ax.tick_params(axis="y", labelcolor="#276FBF")
    ax.spines["top"].set_visible(False)
    other = ax.twinx()
    evaluations = other.plot(
        x,
        generations["Evaluated"].cumsum(),
        marker="s",
        color="#D1495B",
        label="Cumulative evaluations",
    )
    other.set_ylabel("Cumulative completed evaluations", color="#D1495B")
    other.tick_params(axis="y", labelcolor="#D1495B")
    other.spines["top"].set_visible(False)
    total = int(generations["Evaluated"].sum())
    ax.set_title(
        f"Search progress ({total:,} completed evaluations)",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    ax.legend(
        runtime + evaluations,
        [line.get_label() for line in runtime + evaluations],
        frameon=False,
        loc="upper left",
    )


def render_run_figures(
    run_dir: str | PathLike[str],
    output_dir: str | PathLike[str] | None = None,
    *,
    feature_labels: str = "expression",
) -> tuple[Path, ...]:
    """Render all eleven PNGs using only a run's persisted artifacts.

    ``feature_labels`` accepts ``"expression"`` or ``"id"``.
    """

    if feature_labels not in {"expression", "id"}:
        raise ValueError("feature_labels must be 'expression' or 'id'")
    root = Path(run_dir).resolve()
    destination = (
        Path(output_dir).resolve()
        if output_dir is not None
        else root / FIGURES_DIRECTORY
    )
    destination.mkdir(parents=True, exist_ok=True)

    candidates = pd.DataFrame(read_candidates_csv(root / CANDIDATES_FILENAME))
    generations = pd.DataFrame(read_generations_csv(root / GENERATIONS_FILENAME))
    evaluation = read_final_evaluation_tables(root)
    features = pd.DataFrame(evaluation.features)
    importances = pd.DataFrame(evaluation.importances)
    correlations = pd.DataFrame(evaluation.correlations)

    candidates = _numeric(
        candidates, ("Generation", "MaterializationTime", *_FOLD_COLUMNS)
    )
    generations = _numeric(
        generations,
        (
            "Generation",
            "ArchiveSize",
            "Added",
            "Evaluated",
            "CumulativeRuntimeSeconds",
        ),
    )
    features = _numeric(features, _FINAL_FOLD_COLUMNS)
    importances = _numeric(importances, ("importance_mean", "importance_std"))
    correlations = _numeric(correlations, ("spearman", "training_row_count"))

    persisted_metric = evaluation.metrics.get(SEARCH_FOLD_METRIC_KEY)
    metric = persisted_metric if isinstance(persisted_metric, str) else None
    metric_label = metric_display_name(metric)
    evaluation_free = _is_evaluation_free(candidates)
    label_column = "feature_label" if feature_labels == "expression" else "feature_id"
    row_count_value = evaluation.metrics.get("correlation_training_row_count")
    row_count = (
        int(row_count_value)
        if isinstance(row_count_value, (int, float))
        and math.isfinite(float(row_count_value))
        else None
    )

    renderers: tuple[Callable[[plt.Axes], None], ...] = (
        lambda ax: _candidate_outcomes(ax, candidates),
        lambda ax: _archive_changes(ax, generations),
        lambda ax: _score_by_generation(ax, candidates, metric_label, evaluation_free),
        lambda ax: _fold_stability_plot(ax, candidates, metric_label, evaluation_free),
        lambda ax: _materialization_time(ax, candidates),
        lambda ax: _score_vs_time(ax, candidates, metric_label, evaluation_free),
        lambda ax: _final_fold_scores(
            ax, features, metric_label, evaluation_free, label_column
        ),
        lambda ax: _importance(ax, importances, label_column),
        lambda ax: _heatmap(ax, correlations, importances, label_column, row_count),
        lambda ax: _primitive_presence(ax, features),
        lambda ax: _runtime(ax, generations),
    )
    paths: list[Path] = []
    for filename, renderer in zip(FIGURE_FILENAMES, renderers, strict=True):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        renderer(ax)
        path = destination / filename
        _save(fig, path)
        paths.append(path)
    return tuple(paths)


__all__ = [
    "FIGURE_DPI",
    "FIGURE_FILENAMES",
    "FIGURES_DIRECTORY",
    "fold_stability",
    "generation_iqr",
    "metric_display_name",
    "render_run_figures",
]
