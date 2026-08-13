"""Create strategy diagnostic plots from automatedfe search run results.

Consumes only the outputs already produced by scripts/search.py:

  runs/<run_group>/seed_<seed>/summary.json
  runs/<run_group>/seed_<seed>/diagnostics.csv   (evaluated strategies only)
  runs/<run_group>/seed_<seed>/archive.json      (evaluated strategies only)

or the flat one-off layout produced by ad-hoc runs:

  runs/<stem>.csv + runs/<stem>-summary.json [+ runs/<stem>-archive.json]

The score used is ``final_metrics.roc_auc`` from each summary.json. Metrics
that automatedfe does not record (e.g. final dataset materialisation time)
are rendered as "unavailable" placeholders.
"""

import argparse
import json
import math
import re
import sys
from collections.abc import Callable
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from scipy import stats

MetricSpec = tuple[str, str]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS_DIR = PROJECT_ROOT / "runs"
DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "plots" / "automatedfe_strategy_diagnostics.png"

SEED_DIR_PATTERN = re.compile(r"^seed_(\d+)$")

STRATEGY_LABELS = {
    "genetic": "Standard GP",
    "gp_active": "GP Active",
    "gp_without_active": "Standard GP",
    "genetic_search": "Standard GP",
    "enumerative": "Enumerative Search",
    "enumerative_search": "Enumerative Search",
    "random": "Random Search",
    "random_search": "Random Search",
    "enumerative_without_archive": "Brute Force",
    "unbound_enumerative": "Brute Force",
    "brute_force": "Brute Force",
}
STRATEGY_COLORS = {
    "Standard GP": "tab:blue",
    "GP Active": "tab:cyan",
    "Enumerative Search": "tab:orange",
    "Random Search": "tab:green",
    "Brute Force": "tab:red",
}
STRATEGY_MARKERS = {
    "Standard GP": "o",
    "GP Active": "P",
    "Enumerative Search": "^",
    "Random Search": "D",
    "Brute Force": "*",
}
STRATEGY_LINESTYLES = {
    "Random Search": "--",
}
BRUTE_FORCE_REFERENCE_STRATEGIES = ("Brute Force",)
BRUTE_FORCE_REFERENCE_COLOR = "red"

PLOT_SCORE_COLUMN = "plot_score"
DEFAULT_SCORE_METRIC = "roc_auc"


ARCHIVE_SIZE_METRIC: MetricSpec = ("plot_archive_size", "Archive Size")
FINAL_ARCHIVE_FEATURE_METRIC: MetricSpec = (
    "plot_final_archive_feature_count",
    "Final Archive Feature Count",
)
GENERATION_METRIC: MetricSpec = ("plot_generation_reached", "Generations Reached")
EVALUATIONS_MADE_METRIC: MetricSpec = ("plot_evaluations_made", "Evaluations Made")
SEARCH_SECONDS_PER_CANDIDATE_METRIC = (
    "plot_search_seconds_per_candidate",
    "Search Seconds / Candidate",
)
TIME_METRICS: tuple[MetricSpec, ...] = (
    ("total_time", "Total Time (s)"),
)
FEATURE_METRICS: tuple[MetricSpec, ...] = (
    FINAL_ARCHIVE_FEATURE_METRIC,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create strategy diagnostic plots from automatedfe run results "
            "(summary.json / diagnostics.csv / archive.json)."
        )
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Directory containing run groups or flat result files (default: {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Base output image path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--score-metric",
        type=str,
        default=DEFAULT_SCORE_METRIC,
        help=(
            "Final-metrics key used as the run score "
            f"(default: {DEFAULT_SCORE_METRIC})"
        ),
    )
    parser.add_argument(
        "--stats-output",
        type=Path,
        default=None,
        help=(
            "Output text file for statistical tests. Defaults to a "
            "'*_statistical_tests.txt' file next to the plot."
        ),
    )
    return parser.parse_args()


def clean_label(value: object) -> str:
    text = str(value).strip().replace("-", "_")
    if not text:
        return "Unknown"
    return STRATEGY_LABELS.get(text, text.replace("_", " ").title())


def discover_run_groups(runs_dir: Path) -> list[Path]:
    """Return subdirectories that contain seed_<n> run directories."""
    groups = []
    for child in sorted(runs_dir.iterdir()):
        if not child.is_dir():
            continue
        if any(SEED_DIR_PATTERN.match(path.name) for path in child.iterdir()):
            groups.append(child)
    return groups


def discover_flat_runs(runs_dir: Path) -> list[tuple[str, Path, Path | None]]:
    """Return (strategy_key, summary_path, csv_path) for flat one-off runs.

    Flat runs are named <stem>-summary.json with an optional <stem>.csv
    diagnostics file, e.g. genetic-60s-summary.json + genetic-60s.csv.
    The strategy key is the longest known strategy prefix of the stem.
    """
    strategy_prefixes = (
        "enumerative_without_archive",
        "enumerative-without-archive",
        "enumerative",
        "unbound_enumerative",
        "brute_force",
        "genetic",
        "random",
    )
    runs: list[tuple[str, Path, Path | None]] = []
    for summary_path in sorted(runs_dir.glob("*-summary.json")):
        stem = summary_path.name[: -len("-summary.json")]
        strategy_key = next(
            (prefix for prefix in strategy_prefixes if stem.startswith(prefix)),
            stem.split("-")[0],
        )
        csv_path = runs_dir / f"{stem}.csv"
        runs.append((strategy_key, summary_path, csv_path if csv_path.exists() else None))
    return runs


def read_summary(summary_path: Path) -> dict[str, object]:
    with open(summary_path, encoding="utf-8") as f:
        return json.load(f)


def read_diagnostics(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    for column in ("Generation", "MaterializationTime"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def diagnostics_aggregates(csv_path: Path | None) -> dict[str, float]:
    """Per-run diagnostics aggregates: generation, materialisation time, features."""
    if csv_path is None or not csv_path.exists():
        return {
            "plot_generation_reached": np.nan,
            "feature_materialization_time": np.nan,
            "num_features_materialized": np.nan,
        }
    try:
        df = read_diagnostics(csv_path)
    except (OSError, pd.errors.ParserError, ValueError):
        return {
            "plot_generation_reached": np.nan,
            "feature_materialization_time": np.nan,
            "num_features_materialized": np.nan,
        }

    generation = (
        float(df["Generation"].max())
        if "Generation" in df.columns and not df["Generation"].dropna().empty
        else np.nan
    )
    materialization_time = (
        float(df["MaterializationTime"].sum())
        if "MaterializationTime" in df.columns and not df["MaterializationTime"].dropna().empty
        else np.nan
    )

    materialized_features = np.nan
    if "Dependencies" in df.columns:
        unique_features: set[str] = set()
        for dependencies in df["Dependencies"].dropna():
            unique_features.update(
                feature
                for feature in str(dependencies).split(";")
                if feature.strip()
            )
        materialized_features = float(len(unique_features))

    return {
        "plot_generation_reached": generation,
        "feature_materialization_time": materialization_time,
        "num_features_materialized": materialized_features,
    }


def as_float(value: object, default: float = np.nan) -> float:
    try:
        return float(value) if value is not None else default
    except (TypeError, ValueError):
        return default


def run_row(
    summary_path: Path,
    csv_path: Path | None,
    seed: object,
    strategy_label: str,
    run_group: str,
    score_metric: str,
) -> dict[str, object]:
    summary = read_summary(summary_path)
    configuration = summary.get("configuration", {})
    final_metrics = summary.get("final_metrics", {})
    counts = summary.get("counts", {})
    timings = summary.get("timings", {})

    seed_value = as_float(configuration.get("seed"), np.nan)
    if np.isnan(seed_value):
        seed_value = as_float(seed, np.nan)

    objectives = summary.get("objectives")
    objective_count = float(len(objectives)) if isinstance(objectives, list) else np.nan
    full_archive_count = as_float(summary.get("full_archive_feature_count"), objective_count)
    selected_count = as_float(summary.get("selected_feature_count"), objective_count)

    search_seconds = as_float(timings.get("search_seconds"))
    final_evaluation_seconds = as_float(timings.get("final_evaluation_seconds"))
    evaluations = as_float(counts.get("evaluated"))
    score = as_float(final_metrics.get(score_metric))

    row = {
        "seed": seed_value,
        "run_group": run_group,
        "strategy_label": strategy_label,
        "plot_score": score,
        "plot_archive_size": full_archive_count,
        "plot_final_archive_feature_count": selected_count,
        "plot_evaluations_made": evaluations,
        "plot_candidates_evaluated": evaluations,
        "search_time": search_seconds,
        "final_rf_fit_time": final_evaluation_seconds,
        "total_time": search_seconds + final_evaluation_seconds,
        # automatedfe does not record this separately:
        "final_dataset_materialization_time": np.nan,
        "plot_search_seconds_per_candidate": (
            search_seconds / evaluations if evaluations > 0 else np.nan
        ),
    }
    row.update(diagnostics_aggregates(csv_path))
    return row


def seed_run_row(
    run_dir: Path,
    strategy_label: str,
    run_group: str,
    score_metric: str,
) -> dict[str, object]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {}
    seed_match = SEED_DIR_PATTERN.match(run_dir.name)
    seed = int(seed_match.group(1)) if seed_match else None
    return run_row(
        summary_path,
        run_dir / "diagnostics.csv",
        seed,
        strategy_label,
        run_group,
        score_metric,
    )


def load_results(runs_dir: Path | str, score_metric: str) -> tuple[pd.DataFrame, list[str]]:
    runs_dir = Path(runs_dir)
    rows: list[dict[str, object]] = []
    warnings: list[str] = []

    for group_dir in discover_run_groups(runs_dir):
        run_group = group_dir.name
        strategy_label = clean_label(run_group)
        for run_dir in sorted(group_dir.iterdir()):
            row = seed_run_row(run_dir, strategy_label, run_group, score_metric)
            if row:
                rows.append(row)

    for strategy_key, summary_path, csv_path in discover_flat_runs(runs_dir):
        strategy_label = clean_label(strategy_key)
        row = run_row(
            summary_path,
            csv_path,
            None,
            strategy_label,
            strategy_key,
            score_metric,
        )
        if row:
            rows.append(row)

    if not rows:
        raise ValueError(f"No run results found under {runs_dir}.")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["seed"]).copy()
    df["seed"] = df["seed"].astype(int)
    return df, warnings


def strategy_order(df: pd.DataFrame) -> list[str]:
    preferred = list(STRATEGY_COLORS)
    present = list(dict.fromkeys(df["strategy_label"].tolist()))
    return [label for label in preferred if label in present] + [
        label for label in present if label not in preferred
    ]


def color_for(label: str) -> str:
    return STRATEGY_COLORS.get(label, f"C{abs(hash(label)) % 10}")


def marker_for(label: str) -> str:
    return STRATEGY_MARKERS.get(label, "o")


def linestyle_for(label: str) -> str:
    return STRATEGY_LINESTYLES.get(label, "-")


def is_brute_force_reference(label: str) -> bool:
    return label in BRUTE_FORCE_REFERENCE_STRATEGIES


def comparable_strategy_order(order: list[str]) -> list[str]:
    return [label for label in order if not is_brute_force_reference(label)]


def strategy_plot_order(
    order: list[str],
    strategy_filter: tuple[str, ...] | None = None,
) -> list[str]:
    if strategy_filter is None:
        return comparable_strategy_order(order)
    return [
        label
        for label in comparable_strategy_order(order)
        if label in strategy_filter
    ]


def show_unavailable_axis(ax: plt.Axes, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center")
    ax.set_axis_off()


def brute_force_reference_value(df: pd.DataFrame, metric: str) -> tuple[str, float] | None:
    if metric not in df.columns:
        return None
    for label in BRUTE_FORCE_REFERENCE_STRATEGIES:
        values = (
            pd.to_numeric(
                df.loc[df["strategy_label"] == label, metric],
                errors="coerce",
            )
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values):
            return label, float(np.median(values))
    return None


def add_brute_force_score_reference(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_column: str,
) -> None:
    reference = brute_force_reference_value(df, score_column)
    if reference is None:
        return
    label, value = reference
    ax.axhline(
        value,
        color=BRUTE_FORCE_REFERENCE_COLOR,
        linestyle="--",
        linewidth=1.8,
        alpha=0.9,
        label=f"{label} Upper Bound ({value:.4f})",
        zorder=2,
    )


def metric_frame(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    if metric not in df.columns:
        return pd.DataFrame(columns=df.columns)
    return df.dropna(subset=[metric])


def filtered_metric_frame(
    df: pd.DataFrame,
    metric: str,
    strategy_filter: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    data = metric_frame(df, metric)
    if strategy_filter is not None:
        data = data[data["strategy_label"].isin(strategy_filter)]
    return data


def filtered_complete_frame(
    df: pd.DataFrame,
    columns: list[str],
    strategy_filter: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if any(column not in df.columns for column in columns):
        return pd.DataFrame(columns=df.columns)
    data = df.dropna(subset=columns)
    if strategy_filter is not None:
        data = data[data["strategy_label"].isin(strategy_filter)]
    return data


def plot_metric_by_seed(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    order: list[str],
    strategy_filter: tuple[str, ...] | None = None,
) -> None:
    data = filtered_metric_frame(df, metric, strategy_filter)
    if data.empty:
        show_unavailable_axis(ax, f"No valid {metric}")
        return

    plot_order = strategy_plot_order(order, strategy_filter)
    for label in plot_order:
        strategy_df = data[data["strategy_label"] == label].sort_values("seed")
        if strategy_df.empty:
            continue
        ax.plot(
            strategy_df["seed"],
            strategy_df[metric],
            marker="o",
            linewidth=1.5,
            markersize=3.5,
            label=label,
            color=color_for(label),
        )

    ax.set_title(title)
    ax.set_xlabel("Seed")
    ax.set_ylabel(ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(alpha=0.25)


def plot_scatter(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    x_label: str,
    y_label: str,
    title: str,
    order: list[str],
    strategy_filter: tuple[str, ...] | None = None,
) -> None:
    data = filtered_complete_frame(df, [x_column, y_column], strategy_filter)
    if data.empty and (x_column not in df.columns or y_column not in df.columns):
        show_unavailable_axis(ax, "Missing required columns")
        return

    if data.empty:
        show_unavailable_axis(ax, "No valid points")
        return

    plot_order = strategy_plot_order(order, strategy_filter)
    for label in plot_order:
        strategy_df = data[data["strategy_label"] == label]
        if strategy_df.empty:
            continue
        ax.scatter(
            strategy_df[x_column],
            strategy_df[y_column],
            s=34,
            alpha=0.75,
            color=color_for(label),
            label=label,
            edgecolors="white",
            linewidths=0.4,
        )

    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.25)


def plot_pareto_front(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_column: str,
    order: list[str],
    x_column: str = "feature_materialization_time",
    x_label: str = "Median Feature Materialisation Time (s)\n(Lower is better)",
    title: str = "Pareto Front: Median Materialisation Time vs Median Final Score",
) -> None:
    y_column = score_column
    if x_column not in df.columns or y_column not in df.columns:
        show_unavailable_axis(ax, "Missing required columns")
        return

    data = df.dropna(subset=[x_column, y_column]).copy()
    if data.empty:
        show_unavailable_axis(ax, "No valid points")
        return

    data = (
        data.groupby("strategy_label", as_index=False)
        .agg(
            {
                x_column: "median",
                y_column: "median",
            }
        )
    )
    data = data[~data["strategy_label"].isin(BRUTE_FORCE_REFERENCE_STRATEGIES)]
    if data.empty:
        show_unavailable_axis(ax, "No comparable strategy points")
        add_brute_force_score_reference(ax, df, score_column)
        return

    plot_order = [label for label in order if label in set(data["strategy_label"])]
    for label in plot_order:
        strategy_df = data[data["strategy_label"] == label]
        ax.scatter(
            strategy_df[x_column],
            strategy_df[y_column],
            s=58,
            alpha=0.85,
            color=color_for(label),
            label=label,
            edgecolors="white",
            linewidths=0.7,
        )

    pareto_rows = []
    best_score = -np.inf
    for _, row in data.sort_values([x_column, y_column], ascending=[True, False]).iterrows():
        if row[y_column] > best_score:
            pareto_rows.append(row)
            best_score = row[y_column]
    pareto = pd.DataFrame(pareto_rows).sort_values(x_column, ascending=True)
    ax.plot(
        pareto[x_column],
        pareto[y_column],
        color="black",
        linewidth=2.0,
        label="Pareto Front",
        zorder=3,
    )
    for label in plot_order:
        pareto_strategy_df = pareto[pareto["strategy_label"] == label]
        if pareto_strategy_df.empty:
            continue
        ax.scatter(
            pareto_strategy_df[x_column],
            pareto_strategy_df[y_column],
            s=58,
            alpha=0.85,
            color=color_for(label),
            edgecolors="white",
            linewidths=0.7,
            zorder=4,
        )

    add_brute_force_score_reference(ax, df, score_column)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(score_column)
    ax.grid(alpha=0.25)


def plot_all_runs_pareto_front(
    ax: plt.Axes,
    df: pd.DataFrame,
    score_column: str,
    order: list[str],
    x_column: str = "feature_materialization_time",
) -> None:
    """Plot every independent run, treating enumeration as an ordinary strategy."""
    y_column = score_column
    if x_column not in df.columns or y_column not in df.columns:
        show_unavailable_axis(ax, "Missing required columns")
        return

    data = df.dropna(subset=[x_column, y_column]).copy()
    if data.empty:
        show_unavailable_axis(ax, "No valid points")
        return

    enumeration_reference = brute_force_reference_value(data, y_column)
    data = data[~data["strategy_label"].isin(BRUTE_FORCE_REFERENCE_STRATEGIES)]

    present_strategies = set(data["strategy_label"])
    plot_order = [label for label in order if label in present_strategies]
    for label in plot_order:
        strategy_df = data[data["strategy_label"] == label]
        ax.scatter(
            strategy_df[x_column],
            strategy_df[y_column],
            s=42,
            alpha=0.65,
            color=color_for(label),
            marker=marker_for(label),
            label=label,
            edgecolors="white",
            linewidths=0.5,
            zorder=2,
        )

    medians = (
        data.groupby("strategy_label", as_index=False)[[x_column, y_column]]
        .median()
    )
    median_pareto_rows = []
    best_median_score = -np.inf
    for _, row in medians.sort_values(
        [x_column, y_column],
        ascending=[True, False],
    ).iterrows():
        if row[y_column] > best_median_score:
            median_pareto_rows.append(row)
            best_median_score = row[y_column]

    median_pareto = pd.DataFrame(median_pareto_rows).sort_values(x_column)
    ax.plot(
        median_pareto[x_column],
        median_pareto[y_column],
        color="black",
        linewidth=1.8,
        label="Median Pareto Front",
        zorder=3,
    )
    for label in plot_order:
        strategy_median = medians[medians["strategy_label"] == label]
        if strategy_median.empty:
            continue
        ax.scatter(
            strategy_median[x_column],
            strategy_median[y_column],
            s=125,
            color=color_for(label),
            marker=marker_for(label),
            edgecolors="black",
            linewidths=1.2,
            zorder=4,
        )

    if enumeration_reference is not None:
        _, reference_value = enumeration_reference
        ax.axhline(
            reference_value,
            color=BRUTE_FORCE_REFERENCE_COLOR,
            linestyle="--",
            linewidth=1.8,
            alpha=0.9,
            label=f"Brute Force ({reference_value:.4f})",
            zorder=1,
        )

    ax.set_title("Pareto Front of Median Materialisation Time vs Median Score")
    ax.set_xlabel("Feature Materialisation Time (s)")
    ax.set_ylabel("Median Score")
    ax.grid(alpha=0.25)


def plot_median_materialised_features(ax: plt.Axes, df: pd.DataFrame, order: list[str]) -> None:
    """Scatter median features materialised during search vs median score."""
    x_column = "num_features_materialized"
    y_column = "plot_score"
    data = filtered_complete_frame(df, [x_column, y_column])
    if data.empty:
        show_unavailable_axis(ax, "No valid points")
        return

    medians = (
        data.groupby("strategy_label", as_index=False)[[x_column, y_column]]
        .median()
    )
    plot_order = [label for label in order if label in set(medians["strategy_label"])]
    for label in plot_order:
        strategy_df = medians[medians["strategy_label"] == label]
        if strategy_df.empty:
            continue
        ax.scatter(
            strategy_df[x_column],
            strategy_df[y_column],
            s=90,
            color=color_for(label),
            marker=marker_for(label),
            edgecolors="white",
            linewidths=0.7,
            label=label,
            zorder=2,
        )
        ax.annotate(
            label,
            (float(strategy_df[x_column].iloc[0]), float(strategy_df[y_column].iloc[0])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
        )

    add_brute_force_score_reference(ax, data, y_column)
    ax.set_title("Median Features Materialised During Search vs Median Score")
    ax.set_xlabel("Median Features Materialised During Search")
    ax.set_ylabel("Median Score")
    ax.grid(alpha=0.25)


def plot_violin_box(
    ax: plt.Axes,
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    order: list[str],
    strategy_filter: tuple[str, ...] | None = None,
    reference_score_column: str | None = None,
) -> None:
    if metric not in df.columns:
        show_unavailable_axis(ax, f"No valid {metric}")
        return

    data = filtered_metric_frame(df, metric, strategy_filter)
    plot_order = strategy_plot_order(order, strategy_filter)
    grouped = [
        data.loc[data["strategy_label"] == label, metric].to_numpy()
        for label in plot_order
        if not data.loc[data["strategy_label"] == label, metric].dropna().empty
    ]
    labels = [
        label
        for label in plot_order
        if not data.loc[data["strategy_label"] == label, metric].dropna().empty
    ]
    if not grouped:
        show_unavailable_axis(ax, f"No valid {metric}")
        return

    positions = list(range(1, len(grouped) + 1))
    violin = ax.violinplot(
        grouped,
        positions=positions,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for index, body in enumerate(violin["bodies"]):
        body.set_facecolor(color_for(labels[index]))
        body.set_edgecolor("black")
        body.set_alpha(0.35)

    box = ax.boxplot(
        grouped,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.5},
        boxprops={"facecolor": "white", "edgecolor": "black", "linewidth": 1.0},
        whiskerprops={"color": "black", "linewidth": 1.0},
        capprops={"color": "black", "linewidth": 1.0},
    )
    for patch in box["boxes"]:
        patch.set_alpha(0.8)

    for position, values in zip(positions, grouped, strict=True):
        ax.scatter(
            [position] * len(values),
            values,
            color="black",
            alpha=0.45,
            s=12,
            zorder=3,
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    if reference_score_column is not None and metric == reference_score_column:
        add_brute_force_score_reference(ax, df, reference_score_column)
    ax.grid(axis="y", alpha=0.25)


def default_stats_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_statistical_tests.txt")


def dashboard_output_path(base_output_path: Path, suffix: str) -> Path:
    return base_output_path.with_name(f"{base_output_path.stem}_{suffix}.png")


def dashboard_panel_output_dir(output_path: Path) -> Path:
    return output_path.with_suffix("")


def plot_filename(title: str, index: int) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
    return f"{index:02d}_{slug}.png"


def filter_strategies(df: pd.DataFrame, strategies: tuple[str, ...]) -> pd.DataFrame:
    return df[df["strategy_label"].isin(strategies)].copy()


def p_value_text(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value < 0.001:
        return f"{value:.3e}"
    return f"{value:.4f}"


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni correction (statsmodels-free)."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    if n == 0:
        return np.array([], dtype=float)
    adjusted = np.empty(n, dtype=float)
    running_max = 0.0
    for rank, index in enumerate(np.argsort(p_values)):
        value = min(1.0, max(p_values[index] * (n - rank), running_max))
        adjusted[index] = value
        running_max = value
    return adjusted


def metric_groups_by_strategy(
    df: pd.DataFrame,
    metric: str,
    order: list[str],
) -> list[tuple[str, np.ndarray]]:
    if metric not in df.columns:
        return []

    groups: list[tuple[str, np.ndarray]] = []
    for label in order:
        values = (
            pd.to_numeric(df.loc[df["strategy_label"] == label, metric], errors="coerce")
            .dropna()
            .to_numpy(dtype=float)
        )
        if len(values):
            groups.append((label, values))
    return groups


def pairwise_mann_whitney_results(
    groups: list[tuple[str, np.ndarray]],
) -> list[dict[str, float | str | int]]:
    raw_results: list[dict[str, float | str | int]] = []
    raw_p_values: list[float] = []
    for (left_label, left_values), (right_label, right_values) in combinations(groups, 2):
        try:
            test = stats.mannwhitneyu(left_values, right_values, alternative="two-sided")
            statistic = float(test.statistic)
            p_value = float(test.pvalue)
        except ValueError:
            statistic = np.nan
            p_value = np.nan

        denominator = len(left_values) * len(right_values)
        rank_biserial = (2.0 * statistic / denominator) - 1.0 if denominator and not np.isnan(statistic) else np.nan
        raw_results.append(
            {
                "left": left_label,
                "right": right_label,
                "n_left": len(left_values),
                "n_right": len(right_values),
                "median_left": float(np.median(left_values)),
                "median_right": float(np.median(right_values)),
                "u_statistic": statistic,
                "p_value": p_value,
                "p_adjusted": np.nan,
                "rank_biserial": rank_biserial,
            }
        )
        if not np.isnan(p_value):
            raw_p_values.append(p_value)

    if raw_p_values:
        adjusted = holm_adjust(np.asarray(raw_p_values))
        adjusted_index = 0
        for result in raw_results:
            if not np.isnan(float(result["p_value"])):
                result["p_adjusted"] = float(adjusted[adjusted_index])
                adjusted_index += 1

    return raw_results


def statistical_test_report(df: pd.DataFrame, score_column: str) -> str:
    comparable_df = df[~df["strategy_label"].isin(BRUTE_FORCE_REFERENCE_STRATEGIES)].copy()
    order = strategy_order(comparable_df)
    metrics = (
        (score_column, "Final Score"),
        ("feature_materialization_time", "Feature Materialisation Time (s)"),
        ("total_time", "Total Time (s)"),
        (SEARCH_SECONDS_PER_CANDIDATE_METRIC[0], SEARCH_SECONDS_PER_CANDIDATE_METRIC[1]),
    )
    lines = [
        "AutomatedFE Strategy Statistical Tests",
        "",
        "Tests: Kruskal-Wallis omnibus across strategies; pairwise Mann-Whitney U with Holm-adjusted p-values.",
        "Effect size: rank-biserial correlation, positive when the first strategy tends to have larger values.",
    ]

    for metric, label in metrics:
        lines.extend(["", f"{label} ({metric})", "-" * (len(label) + len(metric) + 3)])
        groups = metric_groups_by_strategy(comparable_df, metric, order)
        if len(groups) < 2:
            lines.append("Not enough strategy groups with valid values.")
            continue

        summary = []
        for group_label, values in groups:
            summary.append(
                f"{group_label}: n={len(values)}, mean={np.mean(values):.6g}, median={np.median(values):.6g}"
            )
        lines.append("Groups: " + " | ".join(summary))

        try:
            omnibus = stats.kruskal(*(values for _, values in groups))
            lines.append(
                "Kruskal-Wallis: "
                f"H={float(omnibus.statistic):.6g}, p={p_value_text(float(omnibus.pvalue))}"
            )
        except ValueError as exc:
            lines.append(f"Kruskal-Wallis: not computed ({exc})")

        lines.append("Pairwise Mann-Whitney U:")
        pairwise_results = pairwise_mann_whitney_results(groups)
        for result in pairwise_results:
            lines.append(
                "  "
                f"{result['left']} vs {result['right']}: "
                f"n=({result['n_left']}, {result['n_right']}), "
                f"median=({float(result['median_left']):.6g}, {float(result['median_right']):.6g}), "
                f"U={float(result['u_statistic']):.6g}, "
                f"p={p_value_text(float(result['p_value']))}, "
                f"p_holm={p_value_text(float(result['p_adjusted']))}, "
                f"rank_biserial={float(result['rank_biserial']):.6g}"
            )

    return "\n".join(lines) + "\n"


def write_statistical_tests(
    df: pd.DataFrame,
    score_column: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = statistical_test_report(df, score_column)
    output_path.write_text(report, encoding="utf-8")
    print(report.rstrip())


def dashboard_summary_metrics(score_column: str) -> list[MetricSpec]:
    return [
        (score_column, "Final Score"),
        *TIME_METRICS,
        *FEATURE_METRICS,
    ]


def create_dashboard_grid(total_panels: int) -> tuple[plt.Figure, list[plt.Axes]]:
    columns = 3
    rows = math.ceil(total_panels / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 7.0, rows * 4.6))
    return fig, list(np.atleast_1d(axes).ravel())


def add_dashboard_scatter_panels(
    axes: list[plt.Axes],
    df: pd.DataFrame,
    score_column: str,
    order: list[str],
) -> int:
    panel_index = 0
    plot_scatter(
        axes[panel_index],
        df,
        "num_features_materialized",
        "feature_materialization_time",
        "Features Materialised",
        "Feature Materialisation Time (s)",
        "Features Materialised vs Materialisation Time",
        order,
    )
    panel_index += 1
    plot_pareto_front(axes[panel_index], df, score_column, order)
    panel_index += 1
    plot_pareto_front(
        axes[panel_index],
        df,
        score_column,
        order,
        x_column="final_dataset_materialization_time",
        x_label="Median Final Dataset Materialisation Time (s)\n(Lower is Better)",
        title="Pareto Front: Final Dataset Materialisation Time vs Final Score",
    )
    return panel_index + 1


def add_dashboard_distribution_panels(
    axes: list[plt.Axes],
    start_index: int,
    df: pd.DataFrame,
    score_column: str,
    summary_metrics: list[MetricSpec],
    order: list[str],
) -> int:
    panel_index = start_index
    for metric, label in summary_metrics:
        title = (
            "Final Score Distribution by Strategy"
            if metric == score_column
            else f"{label} Distribution by Strategy"
        )
        plot_violin_box(
            axes[panel_index],
            df,
            metric,
            label,
            title,
            order,
            reference_score_column=score_column,
        )
        panel_index += 1
    return panel_index


def dashboard_note(score_column: str, warnings: list[str]) -> str:
    note = f"Score column: {score_column}"
    if warnings:
        note += " | " + " | ".join(warnings[:2])
        if len(warnings) > 2:
            note += f" | +{len(warnings) - 2} more warnings"
    return note


def create_dashboard(
    df: pd.DataFrame,
    output_path: Path,
    score_column: str,
    warnings: list[str],
    title: str = "AutomatedFE Strategy Diagnostics",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise ValueError(f"No rows available for dashboard: {title}")
    order = strategy_order(df)
    summary_metrics = dashboard_summary_metrics(score_column)
    total_panels = 3 + len(summary_metrics)
    fig, axes_list = create_dashboard_grid(total_panels)
    panel_index = add_dashboard_scatter_panels(
        axes_list,
        df,
        score_column,
        order,
    )
    panel_index = add_dashboard_distribution_panels(
        axes_list,
        panel_index,
        df,
        score_column,
        summary_metrics,
        order,
    )

    for ax in axes_list[panel_index:]:
        ax.set_axis_off()

    handles_by_label = {}
    for ax in axes_list:
        handles, labels = ax.get_legend_handles_labels()
        handles_by_label.update(dict(zip(labels, handles, strict=False)))
    labels = list(handles_by_label)
    handles = [handles_by_label[label] for label in labels]
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=True)

    fig.suptitle(title, fontsize=18, y=0.995)
    fig.text(0.01, 0.006, dashboard_note(score_column, warnings), fontsize=9)
    fig.tight_layout(rect=(0, 0.02, 1, 0.975))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def add_panel_legend(
    fig: plt.Figure,
    ax: plt.Axes,
    location: str | None = None,
    fontsize: float = 8,
) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return

    unique_handles_by_label = dict(zip(labels, handles, strict=False))
    unique_labels = list(unique_handles_by_label)
    unique_handles = [unique_handles_by_label[label] for label in unique_labels]
    if location is not None:
        ax.legend(
            unique_handles,
            unique_labels,
            loc=location,
            frameon=True,
            framealpha=0.9,
            fontsize=fontsize,
        )
        return

    fig.legend(
        unique_handles,
        unique_labels,
        loc="upper center",
        ncol=min(len(unique_labels), 3),
        frameon=True,
        fontsize=fontsize,
    )


def save_dashboard_panel(
    output_dir: Path,
    index: int,
    title: str,
    draw_panel: Callable[[plt.Axes], None],
    legend_location: str | None = None,
    legend_fontsize: float = 8,
) -> Path:
    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.0))
    draw_panel(ax)
    add_panel_legend(fig, ax, legend_location, legend_fontsize)
    top = 1.0 if legend_location is not None else 0.90
    fig.tight_layout(rect=(0, 0, 1, top))

    output_file = output_dir / plot_filename(title, index)
    fig.savefig(output_file, dpi=400, bbox_inches="tight")
    fig.savefig(output_file.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return output_file


def create_dashboard_panel_plots(
    df: pd.DataFrame,
    output_path: Path,
    score_column: str,
) -> list[Path]:
    output_dir = dashboard_panel_output_dir(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        raise ValueError(f"No rows available for dashboard panels: {output_path}")

    order = strategy_order(df)
    panel_specs = [
        (
            "Features Materialized vs Materialization Time",
            lambda ax: plot_scatter(
                ax,
                df,
                "num_features_materialized",
                "feature_materialization_time",
                "Features Materialised",
                "Feature Materialisation Time (s)",
                "Features Materialised vs Materialisation Time",
                order,
            ),
        ),
        (
            "Pareto Front: Median Materialization Time vs Median Final Score",
            lambda ax: plot_pareto_front(ax, df, score_column, order),
        ),
        (
            "Pareto Front: Final Dataset Materialization Time vs Median Final Score",
            lambda ax: plot_pareto_front(
                ax,
                df,
                score_column,
                order,
                x_column="final_dataset_materialization_time",
                x_label="Median Final Dataset Materialisation Time (s)\n(Lower is Better)",
                title="Pareto Front: Final Dataset Materialisation Time vs Final Score",
            ),
        ),
        (
            "Pareto Front: Materialization Time vs Final Score (All Runs)",
            lambda ax: plot_all_runs_pareto_front(ax, df, score_column, order),
        ),
        (
            "Median Features Materialised During Search",
            lambda ax: plot_median_materialised_features(ax, df, order),
        ),
        (
            "Final Score by Seed",
            lambda ax: plot_metric_by_seed(
                ax,
                df,
                score_column,
                score_column,
                "Final Score by Seed",
                order,
            ),
        ),
    ]

    for metric, label in dashboard_summary_metrics(score_column):
        title = (
            "Final Score Distribution by Strategy"
            if metric == score_column
            else f"{label} Distribution by Strategy"
        )
        panel_specs.append(
            (
                title,
                lambda ax, metric=metric, label=label, title=title: plot_violin_box(
                    ax,
                    df,
                    metric,
                    label,
                    title,
                    order,
                    reference_score_column=score_column,
                ),
            )
        )

    output_paths = [
        save_dashboard_panel(
            output_dir,
            index,
            title,
            draw_panel,
            legend_location={
                "Pareto Front: Final Dataset Materialization Time vs Median Final Score": (
                    "lower right"
                ),
                "Pareto Front: Materialization Time vs Final Score (All Runs)": "lower right",
                "Median Features Materialised During Search": "lower right",
            }.get(title),
            legend_fontsize={
                "Pareto Front: Materialization Time vs Final Score (All Runs)": 10,
                "Median Features Materialised During Search": 9,
            }.get(title, 8),
        )
        for index, (title, draw_panel) in enumerate(panel_specs, start=1)
    ]
    return output_paths


def main() -> None:
    args = parse_args()
    df, warnings = load_results(args.runs_dir, args.score_metric)
    for warning in warnings:
        print(warning, file=sys.stderr)
    print(f"Loaded {len(df)} run rows from {args.runs_dir}")
    if df.empty:
        raise ValueError("No run results could be loaded.")
    print("Strategies: " + ", ".join(strategy_order(df)))
    score_column = PLOT_SCORE_COLUMN

    dashboards = (
        (
            "AutomatedFE Strategy Diagnostics: Baseline Comparison",
            filter_strategies(df, tuple(STRATEGY_COLORS)),
            dashboard_output_path(args.output, "baseline_comparison"),
        ),
    )
    for title, dashboard_df, output_path in dashboards:
        create_dashboard(
            dashboard_df,
            output_path,
            score_column,
            warnings,
            title,
        )
        print(f"Saved strategy diagnostics plot to {output_path}")
        panel_paths = create_dashboard_panel_plots(
            dashboard_df,
            output_path,
            score_column,
        )
        print(
            "Saved baseline comparison panel plots to "
            f"{dashboard_panel_output_dir(output_path)} ({len(panel_paths)} files)"
        )

    stats_output = args.stats_output or default_stats_output_path(args.output)
    write_statistical_tests(df, score_column, stats_output)
    print(f"Saved statistical tests to {stats_output}")


if __name__ == "__main__":
    main()
