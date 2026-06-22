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
from statistics import mean, median

from inspect_ai.log import read_eval_log, EvalLog, EvalSample

TASK_TO_DATASET = {
    "abstract_narrative_understanding_task": "abstract_narrative_understanding",
    "agieval_freeform_task": "AGIEval_freeform",
    "agieval_mcq_task": "AGIEval_mcq",
    "bigbenchhard_task": "BigBenchHard",
    "bigtom_task": "BigToM",
    "cause_and_effect_task": "Cause_and_Effect",
    "coqa_task": "CoQA",
    "decompose_task": "LLM_BabyBench_decompose",
    "emobench_task": "EmoBench",
    "evaluating_information_essentiality_task": "Evaluating_Information_Essentiality",
    "ewok_task": "EWoK",
    "fantasy_reasoning_task": "Fantasy_Reasoning",
    "fantom_task": "Fantom",
    "intuit_task": "INTUIT",
    "known_unknowns_task": "Known_Unknowns",
    "macgyver_task": "MacGyver",
    "metamedqa_task": "MetaMedQA",
    "opentom_task": "OpenTOM",
    "plan_bench_task": "Plan_Bench",
    "plan_task": "LLM_BabyBench_plan",
    "predict_task": "LLM_BabyBench_predict",
    "socialnorm_task": "SocialNorm",
    "stepgame_task": "StepGame",
    "text_navigation_task": "Text_Navigation",
    "tiger_mmlu_task": "Tiger_MMLU",
}


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
            "dataset_name": TASK_TO_DATASET.get(task_name, ""),
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
            f, fieldnames=["task_name", "sample_id", "score", "metadata", "dataset_name"]
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {output_path}", file=sys.stderr)


TIMING_FIELDS = [
    "task_name",
    "model",
    "sample_id",
    "total_time",
    "working_time",
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "output_tokens_per_sec",
]

SUMMARY_FIELDS = [
    "task_name",
    "model",
    "n_samples",
    "eval_wall_seconds",
    "total_output_tokens",
    "aggregate_tokens_per_sec",
    "mean_tokens_per_sec",
    "p50_tokens_per_sec",
    "p95_tokens_per_sec",
    "mean_working_time",
    "p50_working_time",
    "p95_working_time",
]


def _sum_usage(model_usage: dict | None, field: str) -> int:
    """Sum a token field across all models in a model_usage mapping."""
    if not model_usage:
        return 0
    return sum(getattr(usage, field, 0) or 0 for usage in model_usage.values())


def _percentile(values: list[float], pct: float) -> float | str:
    """Nearest-rank percentile; returns '' for empty input."""
    if not values:
        return ""
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1))))
    return ordered[rank]


def extract_timing_rows(log: EvalLog) -> list[dict]:
    """Extract per-sample timing and token-usage rows from a log."""
    if log.samples is None:
        return []

    task_name = log.eval.task_display_name or "unknown_task"
    model = log.eval.model
    rows = []

    for sample in log.samples:
        output_tokens = _sum_usage(sample.model_usage, "output_tokens")
        working_time = sample.working_time
        tps = output_tokens / working_time if working_time else ""
        rows.append(
            {
                "task_name": task_name,
                "model": model,
                "sample_id": sample.id,
                "total_time": sample.total_time,
                "working_time": working_time,
                "input_tokens": _sum_usage(sample.model_usage, "input_tokens"),
                "output_tokens": output_tokens,
                "total_tokens": _sum_usage(sample.model_usage, "total_tokens"),
                "output_tokens_per_sec": tps,
            }
        )

    return rows


def summarize_log(log: EvalLog, timing_rows: list[dict]) -> dict:
    """Aggregate one timing summary row per task/log.

    aggregate_tokens_per_sec uses total output tokens over the eval wall-clock,
    which is the correct throughput under concurrency/batching (per-sample
    working_time windows overlap and cannot simply be summed).
    """
    task_name = log.eval.task_display_name or "unknown_task"
    model = log.eval.model

    eval_wall_seconds = ""
    total_output_tokens = ""
    aggregate_tps = ""
    stats = log.stats
    if stats is not None:
        if stats.started_at and stats.completed_at:
            from datetime import datetime

            start = datetime.fromisoformat(stats.started_at)
            end = datetime.fromisoformat(stats.completed_at)
            eval_wall_seconds = (end - start).total_seconds()
        total_output_tokens = _sum_usage(stats.model_usage, "output_tokens")
        if eval_wall_seconds:
            aggregate_tps = total_output_tokens / eval_wall_seconds

    tps_values = [
        r["output_tokens_per_sec"]
        for r in timing_rows
        if isinstance(r["output_tokens_per_sec"], (int, float))
    ]
    wt_values = [
        r["working_time"]
        for r in timing_rows
        if isinstance(r["working_time"], (int, float))
    ]

    return {
        "task_name": task_name,
        "model": model,
        "n_samples": len(timing_rows),
        "eval_wall_seconds": eval_wall_seconds,
        "total_output_tokens": total_output_tokens,
        "aggregate_tokens_per_sec": aggregate_tps,
        "mean_tokens_per_sec": mean(tps_values) if tps_values else "",
        "p50_tokens_per_sec": median(tps_values) if tps_values else "",
        "p95_tokens_per_sec": _percentile(tps_values, 95),
        "mean_working_time": mean(wt_values) if wt_values else "",
        "p50_working_time": median(wt_values) if wt_values else "",
        "p95_working_time": _percentile(wt_values, 95),
    }


def logs_to_timing_csv(
    log_paths: list[str], per_sample_path: str, summary_path: str
) -> None:
    """Extract per-sample timing and per-task summaries to two CSV files."""
    per_sample_rows = []
    summary_rows = []
    expanded_paths = expand_log_paths(log_paths)

    for path_obj in expanded_paths:
        print(f"Timing: {path_obj}", file=sys.stderr)
        log = read_eval_log(str(path_obj))
        rows = extract_timing_rows(log)
        per_sample_rows.extend(rows)
        summary_rows.append(summarize_log(log, rows))

    with open(per_sample_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=TIMING_FIELDS)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    print(
        f"Wrote {len(per_sample_rows)} timing rows to {per_sample_path} "
        f"and {len(summary_rows)} summary rows to {summary_path}",
        file=sys.stderr,
    )


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
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Also extract timing/throughput CSVs (<output>_timing_per_sample.csv "
        "and <output>_timing_summary.csv)",
    )
    args = parser.parse_args()

    logs_to_csv(args.logs, args.output)

    if args.timing:
        stem = Path(args.output).with_suffix("")
        logs_to_timing_csv(
            args.logs,
            f"{stem}_timing_per_sample.csv",
            f"{stem}_timing_summary.csv",
        )


if __name__ == "__main__":
    main()
