"""Portable, atomic HTML reports rendered only from persisted run artifacts."""

from __future__ import annotations

import contextlib
import html
import json
import os
import shutil
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from datetime import UTC, datetime
from os import PathLike
from pathlib import Path
from typing import Final

from .artifacts import (
    CANDIDATES_FILENAME,
    GENERATIONS_FILENAME,
    RUN_MANIFEST_FINGERPRINT_FILENAME,
    RUN_STATUS_FILENAME,
    fingerprint_file,
    load_run_manifest,
    load_status,
    read_candidates_csv,
    read_generations_csv,
)
from .run_plots import (
    FIGURE_FILENAMES,
    FIGURES_DIRECTORY,
    metric_display_name,
    render_run_figures,
)
from .run_tables import (
    CORRELATIONS_FILENAME,
    FEATURES_FILENAME,
    IMPORTANCES_FILENAME,
    METRICS_FILENAME,
    SEARCH_FOLD_METRIC_KEY,
    TIMINGS_FILENAME,
    FinalEvaluationTables,
    read_final_evaluation_tables,
)

REPORT_FILENAME: Final[str] = "report.html"
REPORT_METADATA_FILENAME: Final[str] = "report.json"
REPORT_ASSETS_DIRECTORY: Final[str] = "report-artifacts"
REPORT_FORMAT: Final[str] = "automatedfe-run-report"
REPORT_SCHEMA_VERSION: Final[int] = 1
FEATURE_LABEL_MODES: Final[tuple[str, ...]] = ("id", "expression")

_CSV_LINKS = (
    ("Candidate history", CANDIDATES_FILENAME),
    ("Generation history", GENERATIONS_FILENAME),
    ("Final features", FEATURES_FILENAME),
    ("Forest importances", IMPORTANCES_FILENAME),
    ("Spearman correlations", CORRELATIONS_FILENAME),
    ("Materialization timings", TIMINGS_FILENAME),
)


@dataclass(frozen=True, slots=True)
class _ReportInputs:
    root: Path
    manifest: dict[str, object]
    status: dict[str, object]
    candidates: tuple[dict[str, str], ...]
    generations: tuple[dict[str, str], ...]
    evaluation: FinalEvaluationTables


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _safe_artifact(root: Path, relative: object, name: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"Persisted artifact {name!r} must name a relative file")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(
            f"Persisted artifact {name!r} escapes the run directory"
        ) from error
    if not path.is_file():
        raise FileNotFoundError(f"Persisted artifact {name!r} does not exist: {path}")
    return path


def _validate_persisted_fingerprints(
    root: Path, manifest: Mapping[str, object]
) -> None:
    records = manifest.get("artifact_fingerprints")
    if not isinstance(records, Mapping) or not records:
        raise ValueError("Run manifest has no artifact fingerprint records")
    for relative, record in records.items():
        if not isinstance(relative, str) or not isinstance(record, Mapping):
            raise ValueError("Run manifest artifact fingerprints are malformed")
        path = _safe_artifact(root, relative, relative)
        if fingerprint_file(path) != record.get("fingerprint"):
            raise ValueError(f"Persisted artifact fingerprint mismatch: {relative}")
        expected_bytes = record.get("bytes")
        if expected_bytes is not None and expected_bytes != path.stat().st_size:
            raise ValueError(f"Persisted artifact size mismatch: {relative}")


def _load_report_inputs(run_dir: str | PathLike[str]) -> _ReportInputs:
    """Load only bundle-owned files; source datasets and model state are untouched."""

    root = Path(run_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    manifest = load_run_manifest(root, validate_dataset=False)
    status = load_status(root)
    if status["run_id"] != manifest["run_id"]:
        raise ValueError("Run manifest and status identify different runs")
    if status["state"] != "search_complete":
        raise ValueError("Reports can only be rendered for a completed search")
    if manifest.get("state") not in (None, "search_complete"):
        raise ValueError("Run manifest and status disagree about completion")
    _validate_persisted_fingerprints(root, manifest)

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Run manifest artifacts are malformed")
    if artifacts.get("candidates") != CANDIDATES_FILENAME:
        raise ValueError("Run manifest does not name the canonical candidates table")
    if artifacts.get("generations") != GENERATIONS_FILENAME:
        raise ValueError("Run manifest does not name the canonical generations table")
    evaluation_artifacts = artifacts.get("evaluation")
    if not isinstance(evaluation_artifacts, Mapping):
        raise ValueError("Completed run has no persisted final-evaluation tables")

    candidates = tuple(read_candidates_csv(root / CANDIDATES_FILENAME))
    generations = tuple(read_generations_csv(root / GENERATIONS_FILENAME))
    evaluation = read_final_evaluation_tables(root, evaluation_artifacts)
    return _ReportInputs(
        root=root,
        manifest=dict(manifest),
        status=dict(status),
        candidates=candidates,
        generations=generations,
        evaluation=evaluation,
    )


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _number(value: object, *, digits: int = 4) -> str:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return "Unavailable"
    return f"{converted:,.{digits}f}"


def _fingerprint(value: object) -> str:
    text = str(value or "Unavailable")
    return _escape(text[:18] + "…" + text[-10:] if len(text) > 34 else text)


def _budget_summary(manifest: Mapping[str, object]) -> tuple[str, str]:
    configuration = manifest.get("configuration")
    if not isinstance(configuration, Mapping):
        return "Not recorded in this bundle", "Unavailable"

    if manifest.get("strategy") == "enumerative_without_archive":
        candidate_count = configuration.get("candidate_count")
        if isinstance(candidate_count, int) and not isinstance(candidate_count, bool):
            amount = f"{candidate_count:,} candidate{'' if candidate_count == 1 else 's'}"
        else:
            amount = "Unavailable"
        return amount, "Candidate count · enumerative without archive (no time budget)"

    time_budget = configuration.get("time_budget_seconds")
    if isinstance(time_budget, (int, float)) and not isinstance(time_budget, bool):
        value = Decimal(str(time_budget)).normalize()
        amount = f"{value:,f} second{'' if value == 1 else 's'}"
    else:
        amount = "Unavailable"
    return amount, "Wall-clock search time"


def _metadata_document(
    *, rendered_at_utc: str, feature_labels: str, asset_prefix: str
) -> dict[str, object]:
    return {
        "format": REPORT_FORMAT,
        "schema_version": REPORT_SCHEMA_VERSION,
        "rerendered_at_utc": rendered_at_utc,
        "feature_labels": feature_labels,
        "report": REPORT_FILENAME,
        "figures": [
            f"{asset_prefix}/{FIGURES_DIRECTORY}/{name}" for name in FIGURE_FILENAMES
        ],
        "tables": [path for _, path in _CSV_LINKS],
    }


def _figure_captions(metric_label: str, correlation_rows: int) -> tuple[str, ...]:
    return (
        "Every persisted candidate outcome, grouped by the generation in which it reached a final state.",
        "Archive size and the additions and removals derived from adjacent persisted snapshots.",
        f"Best and median candidate mean {metric_label} by generation. Fold objectives are optimized independently.",
        f"Across-fold stability of {metric_label}; lower spread indicates more consistent fold behavior.",
        "Median and interquartile range of original feature-computation duration. These values are not cache-read latency.",
        f"Candidate mean {metric_label} against original materialization duration.",
        f"The three independently optimized search-fold {metric_label} values for every final feature.",
        "Top 20 final features by mean forest impurity importance; error bars show variation across fitted trees.",
        f"Spearman correlations for the top 20 importance-ranked features, calculated from all {correlation_rows:,} imputed training rows.",
        "Presence counts for primitive families and arithmetic operators in final feature expressions.",
        "Persisted cumulative search runtime and completed evaluations by generation.",
    )


def _metric_rows(metrics: Mapping[str, object]) -> str:
    rows: list[str] = []
    for name, value in metrics.items():
        if name in {
            "format",
            "schema_version",
            SEARCH_FOLD_METRIC_KEY,
            "correlation_training_row_count",
            "total_materialization_seconds",
        }:
            continue
        label = (
            "Held-out ROC AUC" if name == "roc_auc" else name.replace("_", " ").title()
        )
        rows.append(
            f'<div class="metric"><dt>{_escape(label)}</dt><dd>{_number(value)}</dd></div>'
        )
    rows.append(
        '<div class="metric"><dt>Final materialization burden</dt>'
        f"<dd>{_number(metrics.get('total_materialization_seconds'), digits=3)} s</dd></div>"
    )
    return "".join(rows)


def _table_links(inputs: _ReportInputs) -> str:
    counts = {
        CANDIDATES_FILENAME: len(inputs.candidates),
        GENERATIONS_FILENAME: len(inputs.generations),
        FEATURES_FILENAME: len(inputs.evaluation.features),
        IMPORTANCES_FILENAME: len(inputs.evaluation.importances),
        CORRELATIONS_FILENAME: len(inputs.evaluation.correlations),
        TIMINGS_FILENAME: len(inputs.evaluation.timings),
    }
    return "".join(
        '<li><a href="{href}"><span>{label}</span><small>{count:,} rows · CSV</small></a></li>'.format(
            href=_escape(path), label=_escape(label), count=counts[path]
        )
        for label, path in _CSV_LINKS
    )


def _figure_html(metric_label: str, correlation_rows: int, *, asset_prefix: str) -> str:
    captions = _figure_captions(metric_label, correlation_rows)
    return "".join(
        f"""<figure id="figure-{index}">
          <a class="figure-link" href="{_escape(asset_prefix)}/{FIGURES_DIRECTORY}/{_escape(filename)}" aria-label="Open figure {index} at full resolution">
            <img src="{_escape(asset_prefix)}/{FIGURES_DIRECTORY}/{_escape(filename)}" alt="Figure {index}: {_escape(caption)}" loading="lazy" decoding="async">
          </a>
          <figcaption><strong>Figure {index}</strong><span>{_escape(caption)}</span></figcaption>
        </figure>"""
        for index, (filename, caption) in enumerate(
            zip(FIGURE_FILENAMES, captions, strict=True), start=1
        )
    )


def _html_document(
    inputs: _ReportInputs,
    *,
    rendered_at_utc: str,
    feature_labels: str,
    asset_prefix: str,
) -> str:
    manifest = inputs.manifest
    status = inputs.status
    metrics = inputs.evaluation.metrics
    raw_metric = metrics.get(SEARCH_FOLD_METRIC_KEY)
    metric_label = metric_display_name(
        raw_metric if isinstance(raw_metric, str) else None
    )
    correlation_rows = int(metrics["correlation_training_row_count"])
    fingerprints = manifest.get("inputs", {})
    dataset = (
        fingerprints.get("dataset", {}) if isinstance(fingerprints, Mapping) else {}
    )
    mapping = (
        fingerprints.get("mapping", {}) if isinstance(fingerprints, Mapping) else {}
    )
    mmap = (
        fingerprints.get("mmap_manifest", {})
        if isinstance(fingerprints, Mapping)
        else {}
    )
    artifact_records = manifest.get("artifact_fingerprints", {})
    artifact_count = (
        len(artifact_records) if isinstance(artifact_records, Mapping) else 0
    )
    evaluation_free = manifest.get("strategy") == "enumerative_without_archive"
    caveat_unavailable = (
        "This evaluation-free strategy did not compute search-fold scores or search-time materialization diagnostics. "
        "Affected figures display an explicit not-applicable panel."
        if evaluation_free
        else "All search diagnostics shown here derive from persisted candidate and generation tables."
    )
    label_name = (
        "Stable feature IDs" if feature_labels == "id" else "Feature expressions"
    )
    budget_amount, budget_basis = _budget_summary(manifest)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="automatedfe-report-format" content="{REPORT_FORMAT}">
  <meta name="automatedfe-report-schema-version" content="{REPORT_SCHEMA_VERSION}">
  <meta name="automatedfe-rerendered-at-utc" content="{_escape(rendered_at_utc)}">
  <meta name="automatedfe-feature-labels" content="{_escape(feature_labels)}">
  <title>Run {_escape(manifest["run_id"])} · automatedfe</title>
  <style>
    :root {{ color-scheme: light; --ink: #172132; --muted: #596477; --paper: #f5f2ea; --sheet: #fffdf8; --line: #d8d2c5; --navy: #183153; --ochre: #8e5511; --success: #276448; --focus: #b84e24; }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.6 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    ::selection {{ background: #ead3a8; color: #172132; }}
    a {{ color: var(--navy); text-decoration-thickness: 1px; text-underline-offset: .2em; }}
    a:hover {{ color: var(--ochre); }}
    a:focus-visible {{ outline: 3px solid var(--focus); outline-offset: 4px; border-radius: 2px; }}
    .skip {{ position: fixed; left: 1rem; top: -5rem; z-index: 10; padding: .6rem .9rem; background: var(--ink); color: white; }}
    .skip:focus {{ top: 1rem; }}
    .layout {{ display: grid; grid-template-columns: minmax(14rem, 20rem) minmax(0, 1fr); min-height: 100vh; }}
    aside {{ position: sticky; top: 0; align-self: start; height: 100vh; padding: 2.5rem 2rem; background: var(--navy); color: #f4f0e8; }}
    .wordmark {{ margin: 0 0 2.5rem; font: 700 1rem/1.2 ui-monospace, "SFMono-Regular", Consolas, monospace; letter-spacing: .06em; }}
    aside nav a {{ display: block; padding: .48rem 0; color: #e6eaf0; text-decoration: none; border-bottom: 1px solid rgba(255,255,255,.13); }}
    aside nav a:hover {{ color: #f4c984; }}
    .aside-meta {{ position: absolute; bottom: 2rem; left: 2rem; right: 2rem; color: #bcc7d6; font-size: .76rem; }}
    main {{ width: min(100%, 78rem); padding: clamp(2rem, 6vw, 6rem) clamp(1.25rem, 5vw, 5.5rem) 6rem; background: var(--sheet); box-shadow: -18px 0 50px rgba(35, 42, 50, .08); }}
    header {{ padding-bottom: 3.5rem; border-bottom: 2px solid var(--ink); }}
    h1, h2 {{ font-family: Georgia, "Times New Roman", serif; letter-spacing: -.025em; text-wrap: balance; }}
    h1 {{ max-width: 16ch; margin: 0; font-size: clamp(2.9rem, 7vw, 5.8rem); line-height: .98; font-weight: 500; overflow-wrap: anywhere; }}
    h2 {{ margin: 0 0 .8rem; font-size: clamp(1.9rem, 4vw, 3.1rem); line-height: 1.08; font-weight: 500; }}
    p {{ max-width: 72ch; }}
    .lede {{ max-width: 58ch; margin: 1.6rem 0 0; color: var(--muted); font-size: 1.15rem; }}
    .status {{ display: inline-flex; align-items: center; gap: .5rem; margin-top: 2rem; padding: .28rem .72rem; border: 1px solid #8bb39e; border-radius: 999px; color: var(--success); font-weight: 700; font-size: .82rem; }}
    .status::before {{ content: ""; width: .5rem; height: .5rem; border-radius: 50%; background: currentColor; }}
    section {{ padding: 4.5rem 0; border-bottom: 1px solid var(--line); }}
    .section-intro {{ margin: 0 0 2rem; color: var(--muted); }}
    .metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(12rem, 1fr)); margin: 2.25rem 0 0; border-top: 1px solid var(--ink); border-bottom: 1px solid var(--ink); }}
    .metric {{ padding: 1.2rem 1.1rem; border-right: 1px solid var(--line); }}
    .metric:last-child {{ border-right: 0; }}
    .metric dt {{ color: var(--muted); font-size: .76rem; text-transform: uppercase; letter-spacing: .08em; }}
    .metric dd {{ margin: .25rem 0 0; font: 600 1.55rem/1.2 Georgia, serif; font-variant-numeric: tabular-nums; }}
    .facts {{ width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }}
    .facts th, .facts td {{ padding: .75rem 0; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
    .facts th {{ width: 34%; color: var(--muted); font-weight: 500; }}
    code {{ font: .82em/1.5 ui-monospace, "SFMono-Regular", Consolas, monospace; overflow-wrap: anywhere; }}
    .downloads {{ margin: 1.8rem 0 0; padding: 0; list-style: none; border-top: 1px solid var(--ink); }}
    .downloads a {{ display: flex; justify-content: space-between; gap: 1rem; padding: .9rem 0; border-bottom: 1px solid var(--line); text-decoration: none; }}
    .downloads small {{ color: var(--muted); white-space: nowrap; }}
    figure {{ margin: 3.5rem 0 5rem; }}
    .figure-link {{ display: block; aspect-ratio: 1.55; background: white; border: 1px solid var(--line); box-shadow: 0 14px 28px rgba(32, 42, 54, .09); }}
    figure img {{ display: block; width: 100%; height: 100%; object-fit: contain; }}
    figcaption {{ display: grid; grid-template-columns: 6.5rem 1fr; gap: 1rem; padding-top: 1rem; color: var(--muted); font-size: .9rem; }}
    figcaption strong {{ color: var(--ink); text-transform: uppercase; letter-spacing: .07em; }}
    .notes {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 2.5rem 3.5rem; }}
    .note h3 {{ margin: 0 0 .5rem; font: 700 .82rem/1.3 ui-sans-serif, system-ui, sans-serif; text-transform: uppercase; letter-spacing: .08em; color: var(--ochre); }}
    .note p {{ margin: 0; color: var(--muted); }}
    footer {{ padding-top: 3rem; color: var(--muted); font-size: .8rem; }}
    @media (max-width: 760px) {{
      .layout {{ display: block; }}
      aside {{ position: static; height: auto; padding: 1rem 1.25rem; }}
      .wordmark {{ margin: 0 0 .75rem; }}
      aside nav {{ display: flex; gap: 1rem; overflow-x: auto; scrollbar-width: thin; }}
      aside nav a {{ flex: 0 0 auto; border: 0; }}
      .aside-meta {{ display: none; }}
      main {{ padding: 3rem 1.25rem 4rem; box-shadow: none; }}
      section {{ padding: 3.5rem 0; }}
      .metrics, .notes {{ grid-template-columns: 1fr; }}
      .metric {{ border-right: 0; border-bottom: 1px solid var(--line); }}
      .facts {{ display: block; overflow-x: auto; }}
      figcaption {{ grid-template-columns: 1fr; gap: .25rem; }}
      .downloads a {{ align-items: baseline; }}
    }}
    @media print {{ aside {{ display: none; }} .layout {{ display: block; }} main {{ width: 100%; padding: 0; box-shadow: none; }} figure {{ break-inside: avoid; }} }}
    @media (prefers-reduced-motion: reduce) {{ html {{ scroll-behavior: auto; }} }}
  </style>
</head>
<body>
  <!--
  THESIS: A run is auditable evidence, not a dashboard; the report leads with provenance and scientific interpretation.
  OWN-WORLD: Archival paper, navy indexing rail, ochre annotations, ruled tables, and full-width figures.
  STORY: Identify the run, verify its integrity, read its outcomes, inspect every figure, and download complete evidence.
  FIRST VIEWPORT: A fixed navy contents rail frames a large run identity, completion state, and the principal held-out result.
  FORM: Scientific dossier; scoped direct composition; seed key not applicable to this precisely specified report artifact.
  FINISH: unreviewed is unfinished; this build ends with an independent finish-review verdict.
  -->
  <a class="skip" href="#main">Skip to report</a>
  <div class="layout">
    <aside aria-label="Report navigation">
      <p class="wordmark">automatedfe / run report</p>
      <nav>
        <a href="#overview">Overview</a><a href="#configuration">Configuration</a><a href="#integrity">Integrity</a><a href="#tables">Tables</a><a href="#figures">Figures</a><a href="#interpretation">Interpretation</a>
      </nav>
      <p class="aside-meta">Schema {REPORT_SCHEMA_VERSION}<br>Rendered {_escape(rendered_at_utc)}</p>
    </aside>
    <main id="main">
      <header id="overview">
        <h1>Run {_escape(manifest["run_id"])}</h1>
        <p class="lede">A self-contained record of search behavior, final-archive evaluation, feature evidence, and rendering provenance.</p>
        <span class="status">{_escape(status["state"].replace("_", " "))}</span>
        <dl class="metrics">{_metric_rows(metrics)}</dl>
      </header>

      <section id="configuration">
        <h2>Run configuration</h2>
        <p class="section-intro">Only configuration preserved in the structured bundle is reported. Unrecorded settings are not reconstructed.</p>
        <table class="facts"><tbody>
          <tr><th>Strategy</th><td>{_escape(manifest["strategy"])}</td></tr>
          <tr><th>Search budget</th><td>{_escape(budget_amount)}</td></tr>
          <tr><th>Budget basis</th><td>{_escape(budget_basis)}</td></tr>
          <tr><th>Created</th><td>{_escape(manifest["created_at_utc"])}</td></tr>
          <tr><th>Search-fold objective</th><td>{_escape(metric_label)}</td></tr>
          <tr><th>Feature labels</th><td>{_escape(label_name)} (<code>{_escape(feature_labels)}</code>)</td></tr>
          <tr><th>Candidates / generations</th><td>{len(inputs.candidates):,} / {len(inputs.generations):,}</td></tr>
          <tr><th>Final features</th><td>{len(inputs.evaluation.features):,}</td></tr>
          <tr><th>Status updated</th><td>{_escape(status.get("updated_at_utc") or "Unavailable")}</td></tr>
        </tbody></table>
      </section>

      <section id="integrity">
        <h2>Integrity record</h2>
        <p class="section-intro">The report was rendered after validating the bundle manifest checksum and {artifact_count:,} persisted artifact fingerprints. Source paths are not required for rerendering.</p>
        <table class="facts"><tbody>
          <tr><th>Bundle schema</th><td>{_escape(manifest.get("bundle_format", manifest.get("format")))} · version {_escape(manifest.get("bundle_schema_version", manifest.get("schema_version")))}</td></tr>
          <tr><th>Dataset fingerprint</th><td><code>{_fingerprint(dataset.get("fingerprint") if isinstance(dataset, Mapping) else None)}</code></td></tr>
          <tr><th>Mapping fingerprint</th><td><code>{_fingerprint(mapping.get("fingerprint") if isinstance(mapping, Mapping) else None)}</code></td></tr>
          <tr><th>Mmap-manifest fingerprint</th><td><code>{_fingerprint(mmap.get("fingerprint") if isinstance(mmap, Mapping) else None)}</code></td></tr>
          <tr><th>Manifest checksum</th><td><a href="{RUN_MANIFEST_FINGERPRINT_FILENAME}">{RUN_MANIFEST_FINGERPRINT_FILENAME}</a></td></tr>
          <tr><th>Report metadata</th><td><a href="{_escape(asset_prefix)}/{REPORT_METADATA_FILENAME}">{REPORT_METADATA_FILENAME}</a></td></tr>
        </tbody></table>
      </section>

      <section id="tables">
        <h2>Complete evidence tables</h2>
        <p class="section-intro">Figure ranking and top-20 limits do not truncate these downloads. Every persisted feature and correlation remains available.</p>
        <ul class="downloads">{_table_links(inputs)}</ul>
        <p><a href="{METRICS_FILENAME}">Evaluation metrics and analysis metadata · JSON</a></p>
      </section>

      <section id="figures">
        <h2>Scientific figures</h2>
        <p class="section-intro">Eleven standalone 300-DPI PNGs, rerendered from the tables linked above. Select a figure to open its original file.</p>
        {_figure_html(metric_label, correlation_rows, asset_prefix=asset_prefix)}
      </section>

      <section id="interpretation">
        <h2>Interpretation notes</h2>
        <div class="notes">
          <div class="note"><h3>Fold optimization</h3><p>The three search folds are independent optimization objectives. Their values summarize search behavior and must not be treated as repeated held-out test estimates.</p></div>
          <div class="note"><h3>Metric semantics</h3><p>Search-fold values use {_escape(metric_label)}. The final forest result is held-out ROC AUC; Brier improvement is never relabeled as ROC AUC.</p></div>
          <div class="note"><h3>Timing semantics</h3><p>Materialization durations record original combined train/test feature computation. A later cache read can be faster and is not substituted for that scientific burden.</p></div>
          <div class="note"><h3>Correlation scope</h3><p>Spearman correlations use the complete imputed training split ({correlation_rows:,} rows), not a sample and not the held-out test split.</p></div>
          <div class="note"><h3>Held-out evaluation</h3><p>Held-out ROC AUC is calculated once from the final forest. The report contains neither predictions nor serialized model state and does not expose accuracy.</p></div>
          <div class="note"><h3>Unavailable panels</h3><p>{_escape(caveat_unavailable)}</p></div>
        </div>
      </section>

      <footer>Report format <code>{REPORT_FORMAT}</code> · schema {REPORT_SCHEMA_VERSION} · rerendered {_escape(rendered_at_utc)} · labels <code>{_escape(feature_labels)}</code></footer>
    </main>
  </div>
</body>
</html>
"""


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        output.write(text)
        output.flush()
        os.fsync(output.fileno())


def _validate_staged_report(
    staging: Path, metadata: Mapping[str, object], *, asset_prefix: str
) -> None:
    html_path = staging / REPORT_FILENAME
    metadata_path = staging / REPORT_METADATA_FILENAME
    if not html_path.is_file() or not metadata_path.is_file():
        raise ValueError("Staged report is missing HTML or metadata")
    document = html_path.read_text(encoding="utf-8")
    if metadata.get("format") != REPORT_FORMAT:
        raise ValueError("Staged report has an unknown format")
    if metadata.get("schema_version") != REPORT_SCHEMA_VERSION:
        raise ValueError("Staged report has an unsupported schema version")

    required_links = [
        f"{asset_prefix}/{REPORT_METADATA_FILENAME}",
        METRICS_FILENAME,
        RUN_MANIFEST_FINGERPRINT_FILENAME,
        *[path for _, path in _CSV_LINKS],
        *[f"{asset_prefix}/{FIGURES_DIRECTORY}/{name}" for name in FIGURE_FILENAMES],
    ]
    for relative in required_links:
        if relative not in document:
            raise ValueError(f"Staged report does not link {relative!r}")
        figure_prefix = f"{asset_prefix}/{FIGURES_DIRECTORY}/"
        linked = (
            staging / FIGURES_DIRECTORY / relative.removeprefix(figure_prefix)
            if relative.startswith(figure_prefix)
            else None
        )
        if linked is not None:
            if not linked.is_file() or linked.read_bytes()[:8] != b"\x89PNG\r\n\x1a\n":
                raise ValueError(f"Staged report figure is not a valid PNG: {relative}")


def _publish_report(staging: Path, root: Path, *, version: str) -> None:
    """Publish immutable assets, then atomically commit the HTML entry point."""

    versions = root / REPORT_ASSETS_DIRECTORY
    versions.mkdir(exist_ok=True)
    version_dir = versions / version
    if version_dir.exists():
        raise FileExistsError(f"Report artifact version already exists: {version_dir}")
    entry_candidate = root / f".{REPORT_FILENAME}.{uuid.uuid4().hex}.tmp"
    try:
        os.replace(staging, version_dir)
        try:
            os.link(version_dir / REPORT_FILENAME, entry_candidate)
        except OSError:
            shutil.copy2(version_dir / REPORT_FILENAME, entry_candidate)
        # This is the single commit point. Every link in the candidate names
        # immutable assets that were fully installed and validated above.
        os.replace(entry_candidate, root / REPORT_FILENAME)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            entry_candidate.unlink()
        shutil.rmtree(version_dir, ignore_errors=True)
        raise


def render_run_report(
    run_dir: str | PathLike[str],
    *,
    feature_labels: str = "expression",
    rendered_at_utc: str | None = None,
) -> Path:
    """Rerender and atomically publish one complete run report.

    Search, feature materialization, and model evaluation are never imported
    or invoked. The renderer consumes only bundle-owned persisted artifacts.
    """

    if feature_labels not in FEATURE_LABEL_MODES:
        raise ValueError("feature_labels must be 'id' or 'expression'")
    inputs = _load_report_inputs(run_dir)
    timestamp = rendered_at_utc or _utc_now()
    if not isinstance(timestamp, str) or not timestamp.strip():
        raise ValueError("rendered_at_utc must be a non-empty string")

    version = uuid.uuid4().hex
    asset_prefix = f"{REPORT_ASSETS_DIRECTORY}/{version}"
    staging = Path(
        tempfile.mkdtemp(prefix=".report.", suffix=".staging", dir=inputs.root)
    )
    try:
        figures = render_run_figures(
            inputs.root,
            staging / FIGURES_DIRECTORY,
            feature_labels=feature_labels,
        )
        if tuple(path.name for path in figures) != FIGURE_FILENAMES:
            raise ValueError("Figure renderer returned an unexpected artifact set")
        metadata = _metadata_document(
            rendered_at_utc=timestamp.strip(),
            feature_labels=feature_labels,
            asset_prefix=asset_prefix,
        )
        _write_text(
            staging / REPORT_METADATA_FILENAME,
            json.dumps(metadata, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        _write_text(
            staging / REPORT_FILENAME,
            _html_document(
                inputs,
                rendered_at_utc=timestamp.strip(),
                feature_labels=feature_labels,
                asset_prefix=asset_prefix,
            ),
        )
        _validate_staged_report(staging, metadata, asset_prefix=asset_prefix)
        _publish_report(staging, inputs.root, version=version)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return inputs.root / REPORT_FILENAME


render_report = render_run_report
rerender_run_report = render_run_report

__all__ = [
    "FEATURE_LABEL_MODES",
    "REPORT_ASSETS_DIRECTORY",
    "REPORT_FILENAME",
    "REPORT_FORMAT",
    "REPORT_METADATA_FILENAME",
    "REPORT_SCHEMA_VERSION",
    "render_report",
    "render_run_report",
    "rerender_run_report",
]
