"""Atomic assembly and reopening of structured search bundles."""

import json
from pathlib import Path

import pytest

from automatedfe.analysis.artifacts import CANDIDATES_COLUMNS
from automatedfe.analysis.run_bundle import (
    RunBundleValidationError,
    RunBundleWriter,
    load_run_bundle,
    write_run_bundle,
)
from automatedfe.features.grammar import MeanAmount
from automatedfe.search.archive import build_snapshot_document


MAPPING = {
    "status": {"approved": 0},
    "capture_method": {"contactless": 0},
    "payment_method": {"credit": 0},
    "card_brand": {"visa": 0},
    "document_type": {"cpf": 0},
}


def _inputs(tmp_path: Path) -> tuple[Path, Path]:
    dataset = tmp_path / "dataset.parquet"
    dataset.write_bytes(b"dataset")
    mmap_dir = tmp_path / "mmap"
    mmap_dir.mkdir()
    (mmap_dir / "manifest.json").write_text(
        json.dumps({"rows": 0, "columns": {}}), encoding="utf-8"
    )
    return dataset, mmap_dir


def _lifecycle():
    row = {column: "" for column in CANDIDATES_COLUMNS}
    snapshot = build_snapshot_document(
        [MeanAmount(0)],
        [[0.1, 0.2, 0.3, 0.01]],
        minimize=[False, False, False, True],
        mapping_ref={"file": "manifest.json", "source": "run_manifest"},
    )

    class Lifecycle:
        candidate_rows = [row]
        generation_rows = []
        snapshot_documents = ((0, snapshot),)

    return Lifecycle()


def test_complete_bundle_publishes_atomically_and_reopens(tmp_path):
    dataset, mmap_dir = _inputs(tmp_path)
    bundle = write_run_bundle(
        tmp_path / "run",
        strategy="enumerative",
        dataset_path=dataset,
        mapping=MAPPING,
        mmap_dir=mmap_dir,
        lifecycle=_lifecycle(),
    )

    assert bundle.state == "search_complete"
    assert bundle.path == tmp_path / "run"
    assert (bundle.path / "snapshots/generation_000000.json").is_file()
    assert bundle.final_archive is not None
    assert "mapping" not in bundle.snapshots[0]
    assert bundle.manifest["inputs"]["mapping"]["mapping"] == MAPPING
    assert load_run_bundle(bundle.path).state == "search_complete"
    assert not list(tmp_path.glob(".*.staging"))


def test_failed_bundle_is_published_under_partial_without_final_archive(tmp_path):
    dataset, mmap_dir = _inputs(tmp_path)
    bundle = write_run_bundle(
        tmp_path / "run",
        strategy="enumerative",
        dataset_path=dataset,
        mapping=MAPPING,
        mmap_dir=mmap_dir,
        lifecycle=_lifecycle(),
        state="search_failed",
        error="search exploded",
    )

    assert bundle.path == tmp_path / "partial/run"
    assert bundle.state == "search_failed"
    assert bundle.status["error"] == "search exploded"
    assert bundle.final_archive is None
    assert not (tmp_path / "run").exists()


def test_reopening_detects_tampered_named_artifact(tmp_path):
    dataset, mmap_dir = _inputs(tmp_path)
    bundle = write_run_bundle(
        tmp_path / "run",
        strategy="enumerative_without_archive",
        dataset_path=dataset,
        mapping=MAPPING,
        mmap_dir=mmap_dir,
        state="search_complete",
    )
    (bundle.path / "candidates.csv").write_text("corrupt\n", encoding="utf-8")

    with pytest.raises(RunBundleValidationError, match="candidates"):
        load_run_bundle(bundle.path)


def test_failed_write_cleans_only_exact_staging_directory(tmp_path):
    dataset, mmap_dir = _inputs(tmp_path)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep", encoding="utf-8")
    writer = RunBundleWriter(
        tmp_path / "run",
        strategy="enumerative_without_archive",
        dataset_path=dataset,
        mapping=MAPPING,
        mmap_dir=mmap_dir,
    )

    writer.cleanup()

    assert not writer.staging_dir.exists()
    assert (unrelated / "keep.txt").read_text(encoding="utf-8") == "keep"
