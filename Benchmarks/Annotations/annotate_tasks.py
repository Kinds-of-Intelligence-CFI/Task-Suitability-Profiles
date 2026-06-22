


import json
import os
import csv
import glob
import random
import re
from collections import defaultdict
from pathlib import Path
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.model import GenerateConfig
from inspect_ai.solver import system_message

from Benchmarks.Annotations.annotation_agent import annotation_agent

DEFAULT_MODEL = "openai/azure/gpt-4o"
DEFAULT_REQUEST_TIMEOUT = 600
DEFAULT_RETRY_ON_ERROR = 3


def run_annotation_eval(annotation_task, model: str, **kwargs):
    """Run an annotation eval with shared retry/timeout defaults.

    Why: preview/thinking models (notably gemini-3.1-pro-preview) can leave
    requests hung mid-stream with no client-side timeout, blocking the whole
    eval forever. Centralising the defaults here means each benchmark gets the
    same protection without duplicating config across 25 task files.
    """
    return eval(
        annotation_task,
        model=model,
        retry_on_error=DEFAULT_RETRY_ON_ERROR,
        **kwargs,
    )


def sanitize_model_name(model: str) -> str:
    """Replace slashes with double underscores for filesystem safety."""
    return model.replace("/", "__")


def versioned_output_path(base_output_path: str, model: str, timestamp: str) -> str:
    """Compute a versioned output path in an annotations/ subfolder.

    Given base_output_path like '.../BigBenchHard/bigbenchhard_annotations.csv',
    returns '.../BigBenchHard/annotations/bigbenchhard_annotations_openai__azure__gpt-4o_20260317_143022.csv'
    """
    parent_dir = os.path.dirname(base_output_path)
    base_name = os.path.splitext(os.path.basename(base_output_path))[0]
    safe_model = sanitize_model_name(model)
    annotations_dir = os.path.join(parent_dir, "annotations")
    os.makedirs(annotations_dir, exist_ok=True)
    return os.path.join(annotations_dir, f"{base_name}_{safe_model}_{timestamp}.csv")


DEFAULT_COMBINED_ANNOTATIONS_PATH = os.path.join(
    Path(__file__).parent.parent.parent,
    "Suitability", "data", "processed", "annotations.csv",
)


def eligible_canonical_ids(
    dataset_name: str,
    combined_annotations_path: str = DEFAULT_COMBINED_ANNOTATIONS_PATH,
) -> set[str]:
    """Return all sample IDs in the combined annotations CSV for the given dataset.

    The combined ``annotations.csv`` is the ground truth of which items make it
    into the downstream IRT/suitability pipeline. It encodes per-benchmark
    filtering decisions (e.g. items with ``factuality == 1`` are dropped for
    most benchmarks but kept for Crow, Text_Navigation, and
    LLM_BabyBench_decompose where canonical "factuality" doesn't apply).

    Used as the source-of-truth eligible pool for both full
    (``sample_fraction == 1.0``) and partial annotation runs, so every run
    produces a strict subset of the combined annotations.
    """
    if not os.path.exists(combined_annotations_path):
        raise FileNotFoundError(
            f"Combined annotations CSV not found: {combined_annotations_path}. "
            f"Annotation runs require this file to be present."
        )

    ids: set[str] = set()
    with open(combined_annotations_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("dataset name") == dataset_name:
                sid = row.get("sample id")
                if sid:
                    ids.add(sid)

    if not ids:
        raise ValueError(
            f"No rows for dataset {dataset_name!r} found in {combined_annotations_path}. "
            f"Check that dataset_name matches the 'dataset name' column."
        )
    return ids


def select_canonical_subset(
    dataset_name: str,
    num_samples: int,
    seed: int = 42,
    combined_annotations_path: str = DEFAULT_COMBINED_ANNOTATIONS_PATH,
) -> set[str]:
    """Pick a deterministic ID subset from ``eligible_canonical_ids``.

    Sorts the eligible IDs then uses ``random.Random(seed).sample`` to pick
    ``num_samples`` of them. Returns a set so callers can filter datasets
    by membership.
    """
    sorted_ids = sorted(eligible_canonical_ids(dataset_name, combined_annotations_path))
    n = min(num_samples, len(sorted_ids))
    rng = random.Random(seed)
    return set(rng.sample(sorted_ids, k=n))


def target_ids_from_csv(
    csv_path: str,
    num_samples: int | None = None,
    seed: int = 42,
) -> set[str]:
    """Return sample IDs from a per-benchmark annotations CSV.

    Reads all unique ``sample id`` values from ``csv_path`` (the top-level
    gpt-4o reference CSV for a benchmark, e.g.
    ``Benchmarks/Annotated_Benchmarks/BigBenchHard/bigbenchhard_annotations.csv``)
    and returns them as strings.

    If ``num_samples`` is provided, returns a deterministic subset of that size
    using ``random.Random(seed).sample`` over the sorted ID list (same
    convention as :func:`select_canonical_subset`).

    Used as the source-of-truth eligible pool for new model annotation runs:
    every sample gpt-4o annotated is offered to the new model, including items
    gpt-4o flagged as low-factuality. Filtering on factuality (or any other
    dimension) is handled downstream.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Annotation CSV not found: {csv_path}. "
            f"The gpt-4o reference CSV must be present to determine which IDs to annotate."
        )

    ids: set[str] = set()
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = row.get("sample id")
            if sid:
                ids.add(str(sid))

    if not ids:
        raise ValueError(f"No sample IDs found in {csv_path}.")

    if num_samples is None:
        return ids

    sorted_ids = sorted(ids)
    n = min(num_samples, len(sorted_ids))
    rng = random.Random(seed)
    return set(rng.sample(sorted_ids, k=n))


DEFAULT_SYSTEM_MESSAGE = """You are an excellent annotation agent that labels benchmark instances using the instructions and rubric (if provided). Your goal is to assign a single integer score that reflects the dimension being evaluated (e.g., capability demand, factuality, ambiguity).

Reason through the instructions, the benchmark instance and the rubric (if provided) before deciding on a score.

When you have completed the task, call the submit() function to report your final answer.
"""


PROMPT_TEMPLATE = """RUBRIC: The following rubric describes six distinct levels of *{dimension}*
required by different tasks:
{rubric}
QUERY: {instance}
ANSWER: {target}
INSTRUCTION: Score the level of *{dimension}* demanded by the given QUERY–ANSWER pair using a discrete value from 0 to 5. Base your judgment on the *{dimension}* needed to
produce the ANSWER in response to the QUERY. Use CHAIN-OF-THOUGHTS
REASONING to reason step by step before assigning the score. After the
CHAIN-OF-THOUGHTS REASONING STEPS, conclude your assessment with the
statement: "Thus, the level of *{dimension}* demanded by the given QUERY-ANSWER pair is: SCORE", where 'SCORE' is the integer score you have determined.
"""

FACTUALITY_PROMPT_TEMPLATE = """QUERY: {instance}
ANSWER: {target}
INSTRUCTION: Given the query and an answer above, evaluate the accuracy or
factuality of this answer. Please rate it using a Likert scale from 1 to 5,
where 5 means surely accurate while 1 means surely inaccurate. You may employ
any reasoning strategy before providing your final answer. However, you must
conclude your assessment with the statement: 'Thus, the accuracy of the given
answer is: X', where 'X' is the score you've determined.
"""

AMBIGUITY_PROMPT_TEMPLATE = """QUERY: {instance}
ANSWER: {target}
INSTRUCTION: Given the query and an answer above, evaluate the ambiguity of
this query, independently of the potential difficulty of answering it. That
is, assess whether the query has a single clear interpretation with all
necessary details provided, or if it lacks critical information making it
difficult to answer confidently. Please rate it using a Likert scale from
1 to 5, where 5 means surely unambiguous (very clear) while 1 means surely
ambiguous (very unclear). You may employ any reasoning strategy before
providing your final answer. However, you must conclude your assessment with
the statement: 'Thus, the ambiguity of the given query is: X', where 'X' is
the score you've determined.
"""


def combine_dataset(task_dataset: Dataset, rubric_data) -> Dataset:
    new_samples = []
    for sample in task_dataset:
        print(sample)
        instance = sample.input
        for rubric_dict in rubric_data:
            dimension = rubric_dict["dimension"]
            rubric = rubric_dict["rubric"]
            combined_string = PROMPT_TEMPLATE.format(dimension=dimension, rubric=rubric, instance=instance, target=sample.target)
            new_sample = Sample(
                input=combined_string,
                id=f"{sample.id}_{dimension}",
                metadata={
                    "dimension": dimension,
                    "sample_id": sample.id,
                }
                )
            new_samples.append(new_sample)

        # manually add the ambiguity and factuality questions
        ambiguity_string = AMBIGUITY_PROMPT_TEMPLATE.format(
            instance=instance,
            target=sample.target
        )
        ambiguity_sample = Sample(
            input=ambiguity_string,
            id=f"{sample.id}_ambiguity",
            metadata={
                "dimension": "ambiguity",
                "sample_id": sample.id,
            }
        )
        new_samples.append(ambiguity_sample)
        factuality_string = FACTUALITY_PROMPT_TEMPLATE.format(
            instance=instance,
            target=sample.target
        )
        factuality_sample = Sample(
            input=factuality_string,
            id=f"{sample.id}_factuality",
            metadata={
                "dimension": "factuality",
                "sample_id": sample.id,
            }
        )
        new_samples.append(factuality_sample)
    return MemoryDataset(new_samples, name=f"{task_dataset.name}_annotation")

# Conclusion statement required by every prompt template, e.g.
# "Thus, the level of *X* ... is: 2" / "the accuracy ... is: 4". Optional
# markdown bold around the number ("is: **2**") is tolerated.
_SCORE_IS_RE = re.compile(r"is:\s*\**\s*(-?\d+)", re.IGNORECASE)
# Some reasoning models emit a JSON block instead, e.g. {"answer": 0}.
_SCORE_JSON_RE = re.compile(r'"answer"\s*:\s*(-?\d+)')


def parse_score(raw) -> str:
    """Extract the integer annotation score from a model completion.

    The annotation agent's submit() tool returns a bare integer, but reasoning
    models (notably gemini-3-flash-preview) frequently skip the tool call and
    emit free text. The prompt instructs them to conclude with
    "Thus, the level of *X* ... is: SCORE" (or accuracy/ambiguity variants), and
    some emit a JSON '{"answer": N}' block instead. This recovers the final
    integer from any of those forms, returning "" when nothing is parseable.
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if s == "":
        return ""
    # Bare integer: submit() succeeded and overwrote the completion.
    if re.fullmatch(r"-?\d+", s):
        return s
    # Conclusion statement: take the LAST match so the final verdict wins over
    # any intermediate "is: ..." phrasing in the reasoning.
    matches = _SCORE_IS_RE.findall(s)
    if matches:
        return matches[-1]
    # JSON answer block.
    m = _SCORE_JSON_RE.search(s)
    if m:
        return m.group(1)
    # Last resort: the final standalone integer in the text.
    nums = re.findall(r"-?\d+", s)
    if nums:
        return nums[-1]
    return ""


def extract_annotations(log: EvalLog, output_file: str, mode: str = "overwrite"):
    """
    Extract annotations from evaluation log and write to CSV file.

    Args:
        log: Evaluation log containing annotation results
        output_file: Path to output CSV file
        mode: "overwrite" to replace file completely, "append" to merge with existing annotations
    """
    assert log.samples is not None

    # Extract new annotations from the log
    new_annotations = []
    for sample in log.samples:
        score = parse_score(sample.output.completion)
        annotation = (log.eval.dataset.name, sample.metadata["sample_id"], sample.metadata["dimension"], score)
        new_annotations.append(annotation)

    if mode == "overwrite" or not os.path.exists(output_file):
        # Overwrite mode or file doesn't exist - write all new annotations
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["dataset name", "sample id", "dimension", "score"])
            writer.writerows(new_annotations)
    elif mode == "append":
        # Append mode - merge with existing annotations
        existing_annotations = []

        # Read existing annotations if file exists
        if os.path.exists(output_file):
            with open(output_file, 'r', newline='', encoding='utf-8') as csvfile:
                reader = csv.reader(csvfile)
                headers = next(reader, None)  # Skip header row
                if headers:
                    for row in reader:
                        if len(row) >= 4:  # Ensure row has all required columns
                            existing_annotations.append(tuple(row))

        # Create a set of existing (sample_id, dimension) pairs to avoid duplicates
        existing_keys = set((row[1], row[2]) for row in existing_annotations)

        # Only add new annotations that don't already exist
        filtered_new_annotations = []
        for annotation in new_annotations:
            key = (annotation[1], annotation[2])  # (sample_id, dimension)
            if key not in existing_keys:
                filtered_new_annotations.append(annotation)

        # Combine existing and new annotations
        all_annotations = existing_annotations + filtered_new_annotations

        # Write combined annotations
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["dataset name", "sample id", "dimension", "score"])
            writer.writerows(all_annotations)

        print(f"Appended {len(filtered_new_annotations)} new annotations to {output_file}")
        if len(filtered_new_annotations) < len(new_annotations):
            skipped = len(new_annotations) - len(filtered_new_annotations)
            print(f"Skipped {skipped} duplicate annotations")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'overwrite' or 'append'")


def discover_annotation_files(
    evaluations_dir: str,
    model: str | None = None,
    timestamp: str | None = None,
    top_level_only: bool = False,
) -> list[str]:
    """Discover annotation CSVs under ``evaluations_dir``.

    Args:
        evaluations_dir: Directory containing benchmark subdirectories
            (e.g. 'Benchmarks/Annotated_Benchmarks').
        model: If set, only include files whose basename ends with
            ``_{sanitize_model_name(model)}_{timestamp}.csv``.
        timestamp: Required when ``model`` is set.
        top_level_only: If True, only include files directly inside each
            benchmark directory (i.e. ``BENCHMARK/{benchmark}_annotations.csv``)
            and exclude any per-run files in ``BENCHMARK/annotations/`` subdirs.
            This is how the original gpt-4o reference annotations are stored.

    Returns:
        Sorted list of CSV paths matching the filters.
    """
    if model is not None and timestamp is None:
        raise ValueError("timestamp must be provided when model is set")
    if model is not None and top_level_only:
        raise ValueError("model/timestamp filter is incompatible with top_level_only")

    if top_level_only:
        # Non-recursive: only files one level below evaluations_dir, not in annotations/ subdirs.
        # The top-level files are the gpt-4o reference set and have no model suffix,
        # so the strict "_annotations.csv" pattern is correct here.
        pattern = os.path.join(evaluations_dir, "*", "*_annotations.csv")
        csv_files = glob.glob(pattern)
    else:
        # Recursive: include both the top-level "_annotations.csv" files and the
        # per-run files in annotations/ subdirs, which end in
        # "_annotations_{safe_model}_{timestamp}.csv".
        pattern = os.path.join(evaluations_dir, "**", "*_annotations*.csv")
        csv_files = glob.glob(pattern, recursive=True)

    if model is not None:
        safe_model = sanitize_model_name(model)
        required_suffix = f"_{safe_model}_{timestamp}.csv"
        csv_files = [p for p in csv_files if os.path.basename(p).endswith(required_suffix)]

    return sorted(csv_files)


def combine_annotations(
    evaluations_dir: str,
    output_path: str,
    exclude_datasets: set[str] | None = None,
    model: str | None = None,
    timestamp: str | None = None,
    top_level_only: bool = False,
    csv_files: list[str] | None = None,
) -> str | None:
    """Combine per-benchmark long-format annotation CSVs into one wide-format CSV.

    Args:
        evaluations_dir: Path to the directory containing benchmark subdirectories
            (e.g. 'Benchmarks/Annotated_Benchmarks'). Ignored when ``csv_files``
            is provided.
        output_path: Where to write the combined wide-format CSV.
        exclude_datasets: Optional set of benchmark directory names to skip.
        model: If provided, only include CSVs whose filename ends with
            ``_{sanitize_model_name(model)}_{timestamp}.csv``. Filters out
            files from other model runs and the top-level gpt-4o
            annotations files (which have no model suffix).
        timestamp: Required when ``model`` is provided. Identifies a specific
            run of that model.
        top_level_only: If True, only include files directly inside each
            benchmark directory and exclude any per-run files in
            ``annotations/`` subdirs. Use this to regenerate the original
            gpt-4o reference combined CSV without mixing in newer per-model runs.
        csv_files: Explicit list of CSV paths to combine. When provided,
            ``evaluations_dir``, ``model``, ``timestamp`` and ``top_level_only``
            are not used to discover files.

    Returns:
        The output_path on success, or None if no annotation data was found.
    """
    rubric_file = os.path.join(Path(__file__).parent, "rubric.json")
    with open(rubric_file, "r") as f:
        rubric_data = json.load(f)
    capability_columns = [entry["dimension"] for entry in rubric_data]

    if csv_files is None:
        csv_files = discover_annotation_files(
            evaluations_dir,
            model=model,
            timestamp=timestamp,
            top_level_only=top_level_only,
        )

    if not csv_files:
        return None

    def benchmark_dir_of(csv_file: str) -> str:
        # Per-model files live in BENCHMARK/annotations/*.csv; top-level files live in BENCHMARK/*.csv.
        parent = os.path.basename(os.path.dirname(csv_file))
        if parent == "annotations":
            return os.path.basename(os.path.dirname(os.path.dirname(csv_file)))
        return parent

    def benchmark_token_of(csv_file: str) -> str:
        # Top-level basename: "{token}_annotations". Per-model: "{token}_annotations_{safe_model}_{timestamp}".
        basename = os.path.splitext(os.path.basename(csv_file))[0]
        if basename.endswith("_annotations"):
            return basename.removesuffix("_annotations")
        idx = basename.find("_annotations_")
        if idx >= 0:
            return basename[:idx]
        return basename

    # Group CSVs by their benchmark directory.
    dir_to_csvs: dict[str, list[str]] = defaultdict(list)
    dir_to_tokens: dict[str, set[str]] = defaultdict(set)
    for csv_file in csv_files:
        benchmark_dir = benchmark_dir_of(csv_file)
        dir_to_csvs[benchmark_dir].append(csv_file)
        dir_to_tokens[benchmark_dir].add(benchmark_token_of(csv_file))

    def dataset_name_for(csv_path: str, benchmark_dir: str) -> str:
        tokens = dir_to_tokens[benchmark_dir]
        if len(tokens) == 1:
            return benchmark_dir
        # Multi-variant benchmark (e.g. AGIEval -> freeform / mcq).
        token = benchmark_token_of(csv_path)
        prefix = benchmark_dir.lower() + "_"
        if token.startswith(prefix):
            variant = token[len(prefix):]
        else:
            variant = token
        return f"{benchmark_dir}_{variant}"

    # Read all CSVs and pivot to wide format
    # Key: (dataset_name, sample_id) -> {dimension: score}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    skip_dimensions = {"ambiguity", "factuality"}

    for benchmark_dir, csv_paths in sorted(dir_to_csvs.items()):
        if exclude_datasets and benchmark_dir in exclude_datasets:
            continue
        for csv_path in sorted(csv_paths):
            ds_name = dataset_name_for(csv_path, benchmark_dir)
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    dimension = row.get("dimension", "")
                    if dimension.lower() in skip_dimensions:
                        continue
                    sample_id = row.get("sample id", "")
                    score = row.get("score", "")
                    key = (ds_name, sample_id)
                    if key not in rows:
                        rows[key] = {}
                    rows[key][dimension] = score

    if not rows:
        return None

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset name", "sample id"] + capability_columns)
        for (ds_name, sample_id) in sorted(rows.keys()):
            dim_scores = rows[(ds_name, sample_id)]
            row_values = [dim_scores.get(col, "") for col in capability_columns]
            writer.writerow([ds_name, sample_id] + row_values)

    return output_path



@task
def annotate_task(
    task_dataset: Dataset,
    message_limit: int | None = None,
):
    assert task_dataset is not None

    if message_limit is None:
        env_limit = os.environ.get("ANNOTATION_MESSAGE_LIMIT")
        if env_limit is not None:
            message_limit = int(env_limit)
    if message_limit is None:
        message_limit = 6  # default limit for annotation conversations

    rubric_data = json.load(open(
        os.path.join(Path(__file__).parent, "./rubric.json"), "r"))
    # now combine rubrics and samples in the template
    annotation_dataset = combine_dataset(task_dataset, rubric_data)

    return Task(
        dataset=annotation_dataset,
        solver=annotation_agent(
            init=system_message(DEFAULT_SYSTEM_MESSAGE),
            message_limit=message_limit,
            token_limit=32000,
        ),
        config=GenerateConfig(timeout=DEFAULT_REQUEST_TIMEOUT),
    )




