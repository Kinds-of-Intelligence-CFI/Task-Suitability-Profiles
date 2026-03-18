"""
Extract data from Inspect AI log files and convert to CSV format.

Usage:
    python extract_logs_to_csv.py --logs log1.json log2.json --output results.csv
    python extract_logs_to_csv.py --logs /path/to/logs/folder --output results.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

from inspect_ai.log import read_eval_log, EvalLog, EvalSample


def sanitize_for_csv(data: dict | None) -> str:
    """Convert metadata dict to JSON string with commas replaced by semicolons."""
    if data is None:
        return ""
    json_str = json.dumps(data)
    return json_str.replace(",", ";")


def convert_score(value) -> str:
    """Convert score value, mapping letter grades to numbers (C->1, I->0)."""
    if value == "C":
        return "1"
    if value == "I":
        return "0"
    return str(value)


def get_primary_score(sample: EvalSample) -> str:
    """Extract the primary (first) score value from a sample."""
    if not sample.scores:
        return ""
    first_scorer = next(iter(sample.scores))
    score = sample.scores[first_scorer]
    return convert_score(score.value)


def extract_log_to_rows(log: EvalLog) -> list[dict]:
    """Extract all samples from a log into row dictionaries."""
    if log.samples is None:
        return []

    task_name = log.eval.task_display_name or "unknown_task"
    rows = []

    for sample in log.samples:
        row = {
            "task_name": task_name,
            "sample_id": sample.id,
            "score": get_primary_score(sample),
            "metadata": sanitize_for_csv(sample.metadata),
        }
        rows.append(row)

    return rows


def expand_log_paths(paths: list[str]) -> list[Path]:
    """Expand paths, converting directories to lists of log files within them."""
    log_extensions = {".json", ".eval"}
    expanded = []

    for path in paths:
        path_obj = Path(path)
        if not path_obj.exists():
            print(f"Warning: Path not found: {path}", file=sys.stderr)
            continue

        if path_obj.is_dir():
            log_files = [
                f for f in path_obj.iterdir()
                if f.is_file() and f.suffix in log_extensions
            ]
            print(f"Found {len(log_files)} log files in {path}", file=sys.stderr)
            expanded.extend(sorted(log_files))
        else:
            expanded.append(path_obj)

    return expanded


def logs_to_csv(log_paths: list[str], output_path: str) -> None:
    """Process multiple log files and write results to CSV."""
    all_rows = []
    expanded_paths = expand_log_paths(log_paths)

    for path_obj in expanded_paths:
        print(f"Processing: {path_obj}", file=sys.stderr)
        log = read_eval_log(str(path_obj))
        rows = extract_log_to_rows(log)
        all_rows.extend(rows)
        print(f"  Extracted {len(rows)} samples", file=sys.stderr)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["task_name", "sample_id", "score", "metadata"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(
        description="Extract data from Inspect AI log files to CSV"
    )
    parser.add_argument(
        "--logs",
        nargs="+",
        required=True,
        help="Paths to log files or directories containing log files",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file path",
    )
    args = parser.parse_args()

    logs_to_csv(args.logs, args.output)


if __name__ == "__main__":
    main()
