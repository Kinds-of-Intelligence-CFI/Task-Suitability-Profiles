


import json
import os
import csv
import glob
from collections import defaultdict
from pathlib import Path
from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.log import EvalLog
from inspect_ai.solver import system_message

from Benchmarks.Annotations.annotation_agent import annotation_agent

DEFAULT_MODEL = "openai/azure/gpt-4o"


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
        score = sample.output.completion
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


def combine_annotations(evaluations_dir: str, output_path: str) -> str | None:
    """Combine per-benchmark long-format annotation CSVs into one wide-format CSV.

    Args:
        evaluations_dir: Path to the directory containing benchmark subdirectories
            (e.g. 'Benchmarks/Annotated_Benchmarks').
        output_path: Where to write the combined wide-format CSV.

    Returns:
        The output_path on success, or None if no annotation data was found.
    """
    rubric_file = os.path.join(Path(__file__).parent, "rubric.json")
    with open(rubric_file, "r") as f:
        rubric_data = json.load(f)
    capability_columns = [entry["dimension"] for entry in rubric_data]

    # Discover annotation CSVs and group by parent directory
    pattern = os.path.join(evaluations_dir, "**", "*_annotations.csv")
    csv_files = glob.glob(pattern, recursive=True)
    if not csv_files:
        return None

    # Group CSVs by their immediate parent directory
    dir_to_csvs: dict[str, list[str]] = defaultdict(list)
    for csv_file in csv_files:
        parent = os.path.basename(os.path.dirname(csv_file))
        dir_to_csvs[parent].append(csv_file)

    # Build dataset name mapping
    def dataset_name_for(csv_path: str, parent_dir: str, sibling_count: int) -> str:
        basename = os.path.splitext(os.path.basename(csv_path))[0]  # e.g. agieval_freeform_annotations
        basename = basename.removesuffix("_annotations")  # e.g. agieval_freeform
        if sibling_count == 1:
            return parent_dir
        # Multi-CSV directory: strip the lowercased dir name prefix to get the suffix
        prefix = parent_dir.lower() + "_"
        if basename.startswith(prefix):
            suffix = basename[len(prefix):]
        else:
            suffix = basename
        return f"{parent_dir}_{suffix}"

    # Read all CSVs and pivot to wide format
    # Key: (dataset_name, sample_id) -> {dimension: score}
    rows: dict[tuple[str, str], dict[str, str]] = {}
    skip_dimensions = {"ambiguity", "factuality"}

    for parent_dir, csv_paths in sorted(dir_to_csvs.items()):
        sibling_count = len(csv_paths)
        for csv_path in sorted(csv_paths):
            ds_name = dataset_name_for(csv_path, parent_dir, sibling_count)
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

    rubric_data = json.load(open(
        os.path.join(Path(__file__).parent, "./rubric.json"), "r"))
    # now combine rubrics and samples in the template
    annotation_dataset = combine_dataset(task_dataset, rubric_data)

    return Task(
        dataset=annotation_dataset,
        solver=annotation_agent(
            init=system_message(DEFAULT_SYSTEM_MESSAGE),
            message_limit=message_limit,
        ),
    )




