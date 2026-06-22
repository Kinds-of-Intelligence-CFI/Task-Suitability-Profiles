"""
Run all benchmark tasks for a given model and combine results.

Usage:
    python run_all_tasks.py --model openai/gpt-4o
    python run_all_tasks.py --model anthropic/claude-3-5-sonnet --max-connections 5
    python run_all_tasks.py --model openai/gpt-4o --include coqa_task bigtom_task
    python run_all_tasks.py --model openai/gpt-4o --dry-run
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from .extract_logs_to_csv import logs_to_csv, logs_to_timing_csv
from .mem_monitor import MemoryMonitor


TASKS = {
    "abstract_narrative_understanding_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Abstract_Narrative_Understanding/abstract_narrartive_understanding_task.py",
        "function": "abstract_narrative_understanding_task",
    },
    "agieval_mcq_task": {
        "file": "Benchmarks/Annotated_Benchmarks/AGIEval/agieval_task.py",
        "function": "agieval_mcq_task",
    },
    "agieval_freeform_task": {
        "file": "Benchmarks/Annotated_Benchmarks/AGIEval/agieval_task.py",
        "function": "agieval_freeform_task",
    },
    "bigbenchhard_task": {
        "file": "Benchmarks/Annotated_Benchmarks/BigBenchHard/bigbenchhard_task.py",
        "function": "bigbenchhard_task",
    },
    "bigtom_task": {
        "file": "Benchmarks/Annotated_Benchmarks/BigToM/bigtom_task.py",
        "function": "bigtom_task",
    },
    "cause_and_effect_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Cause_and_Effect/cause_and_effect_task.py",
        "function": "cause_and_effect_task",
    },
    "coqa_task": {
        "file": "Benchmarks/Annotated_Benchmarks/CoQA/CoQA_task.py",
        "function": "coqa_task",
    },
    "decompose_task": {
        "file": "Benchmarks/Annotated_Benchmarks/LLM_BabyBench/llm_babybench_task.py",
        "function": "decompose_task",
    },
    "emobench_task": {
        "file": "Benchmarks/Annotated_Benchmarks/EmoBench/emobench_task.py",
        "function": "emobench_task",
    },
    "evaluating_information_essentiality_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Evaluating_Information_Essentiality/evaluating_information_essentiality_task.py",
        "function": "evaluating_information_essentiality_task",
    },
    "ewok_task": {
        "file": "Benchmarks/Annotated_Benchmarks/EWoK/ewok_task.py",
        "function": "ewok_task",
    },
    "fantasy_reasoning_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Fantasy_Reasoning/fantasy_reasoning_task.py",
        "function": "fantasy_reasoning_task",
    },
    "fantom_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Fantom/fantom_task.py",
        "function": "fantom_task",
    },
    "intuit_task": {
        "file": "Benchmarks/Annotated_Benchmarks/INTUIT/intuit_task.py",
        "function": "intuit_task",
    },
    "known_unknowns_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Known_Unknowns/known_unknown_task.py",
        "function": "known_unknowns_task",
    },
    "macgyver_task": {
        "file": "Benchmarks/Annotated_Benchmarks/MacGyver/macgyver_task.py",
        "function": "macgyver_task",
    },
    "metamedqa_task": {
        "file": "Benchmarks/Annotated_Benchmarks/MetaMedQA/metamedqa_task.py",
        "function": "metamedqa_task",
    },
    "opentom_task": {
        "file": "Benchmarks/Annotated_Benchmarks/OpenTOM/opentom_task.py",
        "function": "opentom_task",
    },
    "plan_bench_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Plan_Bench/plan_bench_task.py",
        "function": "plan_bench_task",
    },
    "plan_task": {
        "file": "Benchmarks/Annotated_Benchmarks/LLM_BabyBench/llm_babybench_task.py",
        "function": "plan_task",
    },
    "predict_task": {
        "file": "Benchmarks/Annotated_Benchmarks/LLM_BabyBench/llm_babybench_task.py",
        "function": "predict_task",
    },
    "socialnorm_task": {
        "file": "Benchmarks/Annotated_Benchmarks/SocialNorm/socialnorm_task.py",
        "function": "socialnorm_task",
    },
    "stepgame_task": {
        "file": "Benchmarks/Annotated_Benchmarks/StepGame/stepgame_task.py",
        "function": "stepgame_task",
    },
    "text_navigation_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Text_Navigation/text_navigation_task.py",
        "function": "text_navigation_task",
    },
    "tiger_mmlu_task": {
        "file": "Benchmarks/Annotated_Benchmarks/Tiger_MMLU/tiger_mmlu_task.py",
        "function": "tiger_mmlu_task",
    },
}


def get_runnable_tasks() -> dict:
    """Return the mapping of task names to file paths and functions."""
    return TASKS.copy()


def sanitize_model_name(model: str) -> str:
    """Sanitize model name for use in directory paths."""
    return model.replace("/", "_").replace(":", "_")


def create_output_dir(model: str, output_dir: str) -> Path:
    """Create and return the output directory for logs."""
    sanitized_model = sanitize_model_name(model)
    timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    log_dir = Path(output_dir) / sanitized_model / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def run_task(
    task_name: str,
    file_path: str,
    function_name: str,
    model: str,
    log_dir: Path,
    extra_args: list[str],
    retries: int,
    dry_run: bool,
    monitor_memory: bool = False,
) -> tuple[bool, int, float | None, float | None]:
    """
    Run a single task with retry logic.

    When monitor_memory is set, the eval subprocess is run via Popen and its
    process tree is sampled for peak host RAM and GPU VRAM.

    Returns:
        Tuple of (success, attempts, peak_ram_mb, peak_vram_mb). The memory
        values are None when monitoring is disabled or unavailable.
    """
    task_spec = f"{file_path}@{function_name}"
    cmd = [
        "inspect",
        "eval",
        task_spec,
        "--model",
        model,
        "--log-dir",
        str(log_dir),
        *extra_args,
    ]

    if dry_run:
        print(f"[DRY RUN] {' '.join(cmd)}")
        return True, 0, None, None

    peak_ram_mb: float | None = None
    peak_vram_mb: float | None = None

    for attempt in range(1, retries + 1):
        print(f"\n{'='*60}")
        print(f"Running: {task_name} (attempt {attempt}/{retries})")
        print(f"Command: {' '.join(cmd)}")
        print("=" * 60)

        if monitor_memory:
            proc = subprocess.Popen(cmd)
            with MemoryMonitor(proc.pid) as mon:
                returncode = proc.wait()
            peak_ram_mb = mon.peak_ram_mb
            peak_vram_mb = mon.peak_vram_mb
        else:
            returncode = subprocess.run(cmd).returncode

        if returncode == 0:
            print(f"[SUCCESS] {task_name} completed on attempt {attempt}")
            return True, attempt, peak_ram_mb, peak_vram_mb

        if attempt < retries:
            print(f"[RETRY] {task_name} failed, retrying...")

    print(f"[FAILED] {task_name} failed after {retries} attempts")
    return False, retries, peak_ram_mb, peak_vram_mb


def run_all_tasks(
    model: str,
    output_dir: str,
    include: list[str] | None,
    exclude: list[str] | None,
    extra_args: list[str],
    retries: int,
    dry_run: bool,
    timing: bool = False,
) -> Path:
    """
    Run all tasks for the given model.

    Returns:
        Path to the log directory
    """
    tasks = get_runnable_tasks()

    if include:
        unknown = set(include) - set(tasks.keys())
        if unknown:
            print(f"Warning: Unknown tasks will be skipped: {unknown}", file=sys.stderr)
        tasks = {k: v for k, v in tasks.items() if k in include}

    if exclude:
        tasks = {k: v for k, v in tasks.items() if k not in exclude}

    if not tasks:
        print("No tasks to run after filtering.", file=sys.stderr)
        sys.exit(1)

    log_dir = create_output_dir(model, output_dir)
    print(f"\nOutput directory: {log_dir}")
    print(f"Tasks to run: {len(tasks)}")
    print(f"Model: {model}")
    if extra_args:
        print(f"Extra args: {' '.join(extra_args)}")
    print()

    results = {}
    memory_rows = []
    for task_name, task_info in tasks.items():
        success, attempts, peak_ram_mb, peak_vram_mb = run_task(
            task_name=task_name,
            file_path=task_info["file"],
            function_name=task_info["function"],
            model=model,
            log_dir=log_dir,
            extra_args=extra_args,
            retries=retries,
            dry_run=dry_run,
            monitor_memory=timing,
        )
        results[task_name] = {"success": success, "attempts": attempts}
        if timing:
            memory_rows.append(
                {
                    "task_name": task_name,
                    "model": model,
                    "peak_ram_mb": "" if peak_ram_mb is None else round(peak_ram_mb, 1),
                    "peak_vram_mb": ""
                    if peak_vram_mb is None
                    else round(peak_vram_mb, 1),
                }
            )

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    successes = [k for k, v in results.items() if v["success"]]
    failures = [k for k, v in results.items() if not v["success"]]

    print(f"Successful: {len(successes)}/{len(results)}")
    if failures:
        print(f"Failed: {', '.join(failures)}")

    if timing and not dry_run:
        write_memory_csv(log_dir, memory_rows)

    return log_dir


def combine_logs(log_dir: Path) -> None:
    """Combine all logs in the directory into results.csv."""
    output_path = log_dir / "results.csv"
    print(f"\nCombining logs to: {output_path}")
    logs_to_csv([str(log_dir)], str(output_path))


def combine_timing(log_dir: Path) -> None:
    """Extract timing/throughput CSVs from all logs in the directory."""
    per_sample_path = log_dir / "timing_per_sample.csv"
    summary_path = log_dir / "timing_summary.csv"
    print(f"\nExtracting timing to: {summary_path}")
    logs_to_timing_csv([str(log_dir)], str(per_sample_path), str(summary_path))


def write_memory_csv(log_dir: Path, memory_rows: list[dict]) -> None:
    """Write per-task peak RAM/VRAM samples to memory.csv."""
    output_path = log_dir / "memory.csv"
    fieldnames = ["task_name", "model", "peak_ram_mb", "peak_vram_mb"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(memory_rows)
    print(f"Wrote {len(memory_rows)} memory rows to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all benchmark tasks for a given model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all_tasks.py --model openai/gpt-4o
  python run_all_tasks.py --model openai/gpt-4o --max-connections 5
  python run_all_tasks.py --model anthropic/claude-3-5-sonnet --include coqa_task bigtom_task
  python run_all_tasks.py --model openai/gpt-4o --dry-run
  python run_all_tasks.py --model openai/gpt-4o --output-dir my_results
  python run_all_tasks.py --model openai/gpt-4o --retries 5
        """,
    )
    parser.add_argument(
        "--model",
        help="Model identifier (e.g., 'openai/gpt-4o', 'anthropic/claude-3-5-sonnet')",
    )
    parser.add_argument(
        "--output-dir",
        default="logs",
        help="Base output directory (default: logs)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Number of retry attempts on failure (default: 2)",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        help="Only run these specific tasks",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        help="Skip these specific tasks",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )
    parser.add_argument(
        "--no-combine",
        action="store_true",
        help="Skip combining logs to CSV",
    )
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Sample peak RAM/VRAM during each eval and extract throughput "
        "(writes timing_per_sample.csv, timing_summary.csv, memory.csv)",
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="List all available tasks and exit",
    )

    args, extra_args = parser.parse_known_args()

    if args.list_tasks:
        print("Available tasks:")
        for task_name in sorted(get_runnable_tasks().keys()):
            print(f"  {task_name}")
        sys.exit(0)

    if not args.model:
        parser.error("--model is required unless using --list-tasks")

    log_dir = run_all_tasks(
        model=args.model,
        output_dir=args.output_dir,
        include=args.include,
        exclude=args.exclude,
        extra_args=extra_args,
        retries=args.retries,
        dry_run=args.dry_run,
        timing=args.timing,
    )

    if not args.no_combine and not args.dry_run:
        combine_logs(log_dir)
        if args.timing:
            combine_timing(log_dir)


if __name__ == "__main__":
    main()
