"""
evaluation/compare_results.py

Build clean side-by-side comparison tables for a selected set of models.

Example:
    python evaluation/compare_results.py \
        --models word2vec trainword2vec transformer_scratch
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "results"
COMPARISON_DIR = RESULTS_DIR / "comparisons"


TASK_SPECS = {
    "entity_linking": {
        "files": [
            "entity_linking_ncbi.json",
            "entity_linking_bc5cdr_d.json",
            "entity_linking_bc5cdr_c.json",
        ],
        "columns": ["model", "dataset", "acc@1", "acc@5", "acc@10", "mrr"],
        "sort_by": ["dataset", "acc@1"],
        "ascending": [True, False],
    },
    "sts": {
        "files": ["sts_biosses.json"],
        "columns": ["model", "dataset", "pearson_r", "spearman_r"],
        "sort_by": ["pearson_r"],
        "ascending": [False],
    },
    "nli": {
        "files": ["nli_nli4ct.json"],
        "columns": ["model", "dataset", "accuracy", "macro_f1", "majority_baseline"],
        "sort_by": ["accuracy"],
        "ascending": [False],
    },
}


def _load_result(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def _format_value(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    if value is None:
        return "-"
    return str(value)


def _sort_key(value: object) -> tuple[int, object]:
    return (value is None, value)


def _render_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return "No results found for this task."

    widths = {column: len(column) for column in columns}
    formatted_rows: list[dict[str, str]] = []

    for row in rows:
        formatted = {column: _format_value(row.get(column)) for column in columns}
        formatted_rows.append(formatted)
        for column, value in formatted.items():
            widths[column] = max(widths[column], len(value))

    header = " ".join(column.ljust(widths[column]) for column in columns)
    divider = " ".join("-" * widths[column] for column in columns)
    body = [
        " ".join(row[column].ljust(widths[column]) for column in columns)
        for row in formatted_rows
    ]
    return "\n".join([header, divider, *body])


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict], columns: list[str]) -> None:
    with open(path, "w") as handle:
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join("---" for _ in columns) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(_format_value(row.get(column)) for column in columns) + " |\n")


def build_table(models: list[str], task_name: str) -> list[dict]:
    spec = TASK_SPECS[task_name]
    rows: list[dict] = []

    for model in models:
        model_dir = RESULTS_DIR / model
        for filename in spec["files"]:
            result = _load_result(model_dir / filename)
            if result is None:
                continue

            row = {column: result.get(column) for column in spec["columns"]}
            row["model"] = model
            row["dataset"] = result.get("dataset", filename.replace(".json", ""))
            rows.append(row)

    if not rows:
        return []

    for column, ascending in reversed(list(zip(spec["sort_by"], spec["ascending"]))):
        rows.sort(key=lambda row: _sort_key(row.get(column)), reverse=not ascending)
    return rows


def save_outputs(models: list[str]) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    for task_name in TASK_SPECS:
        rows = build_table(models, task_name)
        columns = TASK_SPECS[task_name]["columns"]
        csv_path = COMPARISON_DIR / f"{task_name}_comparison.csv"
        md_path = COMPARISON_DIR / f"{task_name}_comparison.md"

        _write_csv(csv_path, rows, columns)
        _write_markdown(md_path, rows, columns)

        print(f"\n{'=' * 60}")
        print(task_name.upper())
        print(f"{'=' * 60}")
        print(_render_table(rows, columns))
        print(f"\nsaved: {csv_path}")
        print(f"saved: {md_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=["word2vec", "trainword2vec", "transformer_scratch"],
        help="models to compare",
    )
    args = parser.parse_args()
    save_outputs(args.models)


if __name__ == "__main__":
    main()
