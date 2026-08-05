"""Print null counts and percentages for every column in each dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "loan"


def quote_identifier(identifier: str) -> str:
    """Quote a DuckDB identifier safely."""

    return '"' + identifier.replace('"', '""') + '"'


def dataset_paths(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*.parquet" if recursive else "*.parquet"
    return sorted(path for path in input_dir.glob(pattern) if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing parquet datasets (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include parquet files in subdirectories as well",
    )
    args = parser.parse_args()

    if not args.input_dir.is_dir():
        parser.error(f"Input directory does not exist: {args.input_dir}")

    paths = dataset_paths(args.input_dir, args.recursive)
    if not paths:
        parser.error(f"No parquet datasets found in {args.input_dir}")

    connection = duckdb.connect()
    try:
        print("dataset\tcolumn\trows\tnull_count\tnull_percent")
        for path in paths:
            schema = connection.execute(
                "DESCRIBE SELECT * FROM read_parquet(?)", [str(path)]
            ).fetchall()
            count_expressions = [
                f"count(*) - count({quote_identifier(column[0])})"
                for column in schema
            ]
            query = f"""
                SELECT
                    count(*) AS total_rows,
                    {', '.join(count_expressions)}
                FROM read_parquet(?)
            """
            result = connection.execute(query, [str(path)]).fetchone()
            total_rows = result[0]
            relative_path = path.relative_to(args.input_dir)

            for index, column in enumerate(schema, start=1):
                null_count = result[index]
                null_percent = (
                    100.0 * null_count / total_rows if total_rows else 0.0
                )
                print(
                    f"{relative_path}\t{column[0]}\t{total_rows}\t"
                    f"{null_count}\t{null_percent:.6f}%"
                )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
