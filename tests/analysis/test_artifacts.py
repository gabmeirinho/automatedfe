"""Structured run artifact contract: manifest, fingerprints, feature IDs, schemas."""

import json
from pathlib import Path

import pytest

import automatedfe.analysis.artifacts as artifacts_module
from automatedfe.analysis.artifacts import (
    ARCHIVE_SNAPSHOTS_DIRECTORY,
    CANDIDATES_COLUMNS,
    GENERATIONS_COLUMNS,
    RUN_MANIFEST_FILENAME,
    RUN_MANIFEST_FINGERPRINT_FILENAME,
    RUN_MANIFEST_FORMAT,
    RUN_MANIFEST_SCHEMA_VERSION,
    RUN_STATES,
    STATUS_FORMAT,
    STATUS_SCHEMA_VERSION,
    archive_snapshot_path,
    build_dataset_record,
    build_mapping_record,
    build_mmap_manifest_record,
    build_run_manifest,
    build_status,
    feature_id,
    fingerprint_bytes,
    fingerprint_expression,
    fingerprint_file,
    fingerprint_mapping,
    fingerprint_mmap_manifest,
    generation_from_snapshot_path,
    load_run_manifest,
    load_status,
    read_candidates_csv,
    read_generations_csv,
    write_candidates_csv,
    write_generations_csv,
    write_run_manifest,
    write_status,
)
from automatedfe.features.grammar import Add, CountTotal, MeanAmount, Mul

LABEL_MAPPING = {
    "status": {"approved": 0, "complete": 1, "denied": 2, "others": 3},
    "capture_method": {"contactless": 0, "emv": 1, "pix": 2},
    "payment_method": {"debit": 0, "credit": 1, "null": -1},
    "card_brand": {"mastercard": 0, "visa": 1, "null": -1},
    "document_type": {"cnpj": 0, "cpf": 1, "null": -1},
}

MMAP_MANIFEST = {
    "rows": 100,
    "columns": {
        "amount": {"file": "amount.mmap", "dtype": "float64"},
        "status": {"file": "status.mmap", "dtype": "int32"},
    },
}


def sample_manifest(*, inputs=None, artifacts=None):
    return build_run_manifest(
        run_id="run-123",
        strategy="enumerative",
        created_at_utc="2026-08-13T12:00:00Z",
        inputs=inputs
        if inputs is not None
        else {
            "dataset": {
                "source_path": None,
                "fingerprint": "sha256:" + "a" * 64,
                "bytes": 0,
            },
            "mapping": build_mapping_record(LABEL_MAPPING),
            "mmap_manifest": {
                "source_path": None,
                "fingerprint": fingerprint_mmap_manifest(MMAP_MANIFEST),
                "manifest": MMAP_MANIFEST,
            },
        },
        artifacts=artifacts
        if artifacts is not None
        else {
            "candidates": "candidates.csv",
            "generations": "generations.csv",
            "final_archive": "archive/final.json",
            "archive_snapshots": [],
            "status": "status.json",
        },
    )


def test_fingerprint_bytes_is_deterministic_and_content_sensitive():
    assert fingerprint_bytes(b"payload") == fingerprint_bytes(b"payload")
    assert fingerprint_bytes(b"payload") != fingerprint_bytes(b"payload!")
    assert fingerprint_bytes(b"payload").startswith("sha256:")
    assert len(fingerprint_bytes(b"payload")) == len("sha256:") + 64


def test_fingerprint_file_tracks_content_and_ignores_paths(tmp_path):
    first = tmp_path / "a.bin"
    second = tmp_path / "nested" / "b.bin"
    second.parent.mkdir()
    first.write_bytes(b"same bytes")
    second.write_bytes(b"same bytes")

    assert fingerprint_file(first) == fingerprint_file(second)

    first.write_bytes(b"changed bytes")
    assert fingerprint_file(first) != fingerprint_file(second)


def test_fingerprint_file_rejects_missing_path(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        fingerprint_file(tmp_path / "missing.bin")


def test_mapping_fingerprint_ignores_insertion_order():
    reordered = {
        "document_type": {"cnpj": 0, "cpf": 1, "null": -1},
        "card_brand": {"mastercard": 0, "visa": 1, "null": -1},
        "status": {"approved": 0, "complete": 1, "denied": 2, "others": 3},
        "payment_method": {"debit": 0, "credit": 1, "null": -1},
        "capture_method": {"contactless": 0, "emv": 1, "pix": 2},
    }
    assert fingerprint_mapping(LABEL_MAPPING) == fingerprint_mapping(reordered)

    changed = {**LABEL_MAPPING, "status": {"approved": 0, "complete": 9}}
    assert fingerprint_mapping(changed) != fingerprint_mapping(LABEL_MAPPING)


def test_mmap_manifest_fingerprint_is_content_based():
    assert fingerprint_mmap_manifest(MMAP_MANIFEST) == fingerprint_mmap_manifest(
        dict(MMAP_MANIFEST)
    )
    tampered = {**MMAP_MANIFEST, "rows": 101}
    assert fingerprint_mmap_manifest(tampered) != fingerprint_mmap_manifest(
        MMAP_MANIFEST
    )


def test_identical_expressions_produce_identical_feature_ids():
    first = Add(MeanAmount(0), CountTotal(0))
    second = Add(MeanAmount(0), CountTotal(0))
    other = Mul(MeanAmount(1), CountTotal(1))

    assert feature_id(first) == feature_id(second)
    assert feature_id(first) != feature_id(other)
    assert fingerprint_expression(first) == fingerprint_expression(second)
    assert fingerprint_expression(first) != fingerprint_expression(other)


def test_feature_ids_are_stable_and_prefixed():
    expression = Add(MeanAmount(0), CountTotal(0))
    assert feature_id(expression).startswith("feat_")
    assert feature_id(expression) == "feat_" + fingerprint_expression(expression).split(":", 1)[1][:12]


def test_dataset_record_fingerprint_is_content_based(tmp_path):
    first = tmp_path / "dataset.parquet"
    first.write_bytes(b"dataset bytes")
    record = build_dataset_record(first)

    assert record["fingerprint"] == fingerprint_file(first)
    assert record["bytes"] == len(b"dataset bytes")
    assert record["source_path"] == str(first.resolve())

    first.write_bytes(b"different bytes")
    assert build_dataset_record(first)["fingerprint"] != record["fingerprint"]


def test_mapping_record_stores_content_once_and_fingerprints_it(tmp_path):
    mapping_path = tmp_path / "label_mapping.json"
    mapping_path.write_text(json.dumps(LABEL_MAPPING))

    from_dict = build_mapping_record(LABEL_MAPPING)
    from_path = build_mapping_record(mapping_path)

    assert from_dict["fingerprint"] == fingerprint_mapping(LABEL_MAPPING)
    assert from_dict["mapping"] == LABEL_MAPPING
    assert from_dict["source_path"] is None
    assert from_path["source_path"] == str(mapping_path.resolve())
    assert from_path["mapping"] == from_dict["mapping"]


def test_mmap_manifest_record_reads_and_fingerprints_manifest(tmp_path):
    mmap_dir = tmp_path / "mmap"
    mmap_dir.mkdir()
    (mmap_dir / "manifest.json").write_text(json.dumps(MMAP_MANIFEST))

    record = build_mmap_manifest_record(mmap_dir)

    assert record["fingerprint"] == fingerprint_mmap_manifest(MMAP_MANIFEST)
    assert record["manifest"] == MMAP_MANIFEST


def test_snapshot_paths_encode_and_decode_generations(tmp_path):
    path = archive_snapshot_path(tmp_path, 3)
    assert path == tmp_path / ARCHIVE_SNAPSHOTS_DIRECTORY / "generation_000003.json"
    assert generation_from_snapshot_path(path) == 3
    assert generation_from_snapshot_path(tmp_path / "generation_000003.json") == 3
    assert generation_from_snapshot_path(tmp_path / "other.json") is None
    assert generation_from_snapshot_path(tmp_path / "generation_x.json") is None
    with pytest.raises(ValueError, match="non-negative integer"):
        archive_snapshot_path(tmp_path, -1)


def test_run_manifest_round_trip_preserves_content(tmp_path):
    manifest = sample_manifest()
    run_dir = tmp_path / "run"
    write_run_manifest(run_dir, manifest)

    loaded = load_run_manifest(run_dir)
    assert loaded == manifest
    assert loaded["format"] == RUN_MANIFEST_FORMAT
    assert loaded["schema_version"] == RUN_MANIFEST_SCHEMA_VERSION
    assert loaded["inputs"]["mapping"]["mapping"] == LABEL_MAPPING
    assert loaded["artifacts"]["archive_snapshots"] == []
    assert (run_dir / RUN_MANIFEST_FILENAME).is_file()
    assert (run_dir / RUN_MANIFEST_FINGERPRINT_FILENAME).is_file()


def test_run_manifest_rejects_unsupported_schema_version(tmp_path):
    run_dir = tmp_path / "run"
    write_run_manifest(run_dir, sample_manifest())

    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text())
    data["schema_version"] = 999
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="Unsupported run manifest schema version"):
        load_run_manifest(run_dir)


def test_run_manifest_rejects_tampering(tmp_path):
    run_dir = tmp_path / "run"
    write_run_manifest(run_dir, sample_manifest())

    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text())
    data["strategy"] = "tampered"
    manifest_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_run_manifest(run_dir)


def test_run_manifest_rejects_input_record_tampering(tmp_path):
    run_dir = tmp_path / "run"
    manifest = sample_manifest()
    write_run_manifest(run_dir, manifest)

    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    data = json.loads(manifest_path.read_text())
    data["inputs"]["mapping"]["mapping"]["status"]["approved"] = 7
    tampered_text = json.dumps(data, indent=2, sort_keys=True)
    manifest_path.write_text(tampered_text)
    run_dir.joinpath(RUN_MANIFEST_FINGERPRINT_FILENAME).write_text(
        artifacts_module.fingerprint_bytes(tampered_text.encode("utf-8")) + "\n"
    )

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_run_manifest(run_dir)


def test_run_manifest_rejects_missing_checksum(tmp_path):
    run_dir = tmp_path / "run"
    write_run_manifest(run_dir, sample_manifest())
    run_dir.joinpath(RUN_MANIFEST_FINGERPRINT_FILENAME).unlink()

    with pytest.raises(ValueError, match="missing its checksum record"):
        load_run_manifest(run_dir)


def test_paths_alone_are_not_accepted_as_integrity_evidence(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    manifest_path = run_dir / RUN_MANIFEST_FILENAME
    manifest_path.write_text(json.dumps(sample_manifest()))

    with pytest.raises(ValueError, match="missing its checksum record"):
        load_run_manifest(run_dir)

    (run_dir / RUN_MANIFEST_FINGERPRINT_FILENAME).write_text(
        "sha256:" + "0" * 64 + "\n"
    )
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_run_manifest(run_dir)


def test_load_run_manifest_rejects_directories_without_a_manifest(tmp_path):
    (tmp_path / "candidates.csv").write_text("Strategy\n")
    with pytest.raises(ValueError, match="Not a structured run"):
        load_run_manifest(tmp_path)


def test_run_manifest_validates_dataset_fingerprint_when_requested(tmp_path):
    dataset_path = tmp_path / "dataset.parquet"
    dataset_path.write_bytes(b"dataset bytes")
    run_dir = tmp_path / "run"
    manifest = sample_manifest(
        inputs={
            "dataset": build_dataset_record(dataset_path),
            "mapping": build_mapping_record(LABEL_MAPPING),
            "mmap_manifest": {
                "source_path": None,
                "fingerprint": fingerprint_mmap_manifest(MMAP_MANIFEST),
                "manifest": MMAP_MANIFEST,
            },
        }
    )
    write_run_manifest(run_dir, manifest)
    assert load_run_manifest(run_dir) == manifest

    dataset_path.write_bytes(b"tampered bytes")
    with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
        load_run_manifest(run_dir, validate_dataset=True)
    load_run_manifest(run_dir)


def test_run_manifest_rejects_malformed_json(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / RUN_MANIFEST_FILENAME).write_text("{not json")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_run_manifest(run_dir)


def test_status_round_trip_and_validation(tmp_path):
    run_dir = tmp_path / "run"
    status = build_status(
        run_id="run-123",
        state="running",
        updated_at_utc="2026-08-13T12:00:00Z",
    )
    write_status(run_dir, status)

    loaded = load_status(run_dir)
    assert loaded == status
    assert loaded["format"] == STATUS_FORMAT
    assert loaded["schema_version"] == STATUS_SCHEMA_VERSION

    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_status(tmp_path / "missing")

    for invalid in ("unknown", "completed"):
        with pytest.raises(ValueError, match="Unknown run status state"):
            build_status(run_id="run-123", state=invalid)
    assert RUN_STATES == ("running", "search_failed", "interrupted", "search_complete")


def test_status_rejects_unsupported_schema_version(tmp_path):
    run_dir = tmp_path / "run"
    write_status(run_dir, build_status(run_id="run-123", state="running"))

    status_path = run_dir / "status.json"
    data = json.loads(status_path.read_text())
    data["schema_version"] = 99
    status_path.write_text(json.dumps(data, indent=2, sort_keys=True))

    with pytest.raises(ValueError, match="Unsupported run status schema version"):
        load_status(run_dir)


def test_candidates_csv_round_trips_with_schema_validation(tmp_path):
    rows = [
        {
            "Strategy": "enumerative",
            "CandidateIndex": 0,
            "Generation": "",
            "Expression": "(mean_1d + count_1d)",
            "Dependencies": "amount",
            "Split1": 0.1,
            "Split2": 0.2,
            "Split3": 0.3,
            "MaterializationTime": 0.01,
            "ArchiveMember": True,
            "Status": "evaluated",
            "Error": "",
        }
    ]
    path = tmp_path / "candidates.csv"
    write_candidates_csv(path, rows)

    loaded = read_candidates_csv(path)
    assert list(loaded[0]) == list(CANDIDATES_COLUMNS)
    assert loaded[0]["CandidateIndex"] == "0"
    assert loaded[0]["ArchiveMember"] == "True"

    with pytest.raises(ValueError, match="exactly the schema columns"):
        write_candidates_csv(path, [{"only_one": "column"}])

    tampered = tmp_path / "tampered.csv"
    tampered.write_text("Strategy,Bogus\nx,y\n")
    with pytest.raises(ValueError, match="schema columns"):
        read_candidates_csv(tampered)


def test_generations_csv_round_trips_with_schema_validation(tmp_path):
    rows = [
        {
            "Strategy": "genetic",
            "Generation": 1,
            "Generated": 50,
            "Unique": 48,
            "Duplicate": 2,
            "Invalid": 1,
            "Evaluated": 47,
            "ArchiveSize": 5,
            "Added": 3,
            "DurationSeconds": 1.5,
            "CumulativeRuntimeSeconds": 2.5,
        }
    ]
    path = tmp_path / "generations.csv"
    write_generations_csv(path, rows)

    loaded = read_generations_csv(path)
    assert list(loaded[0]) == list(GENERATIONS_COLUMNS)
    assert loaded[0]["Generation"] == "1"
    assert loaded[0]["Added"] == "3"


def test_generations_csv_rejects_the_obsolete_removed_column(tmp_path):
    path = tmp_path / "historical-generations.csv"
    path.write_text(
        ",".join((*GENERATIONS_COLUMNS, "Removed")) + "\n"
        + ",".join("" for _ in (*GENERATIONS_COLUMNS, "Removed"))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema columns"):
        read_generations_csv(path)
