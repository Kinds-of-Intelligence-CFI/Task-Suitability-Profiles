import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character

from Benchmarks.Annotations.annotate_tasks import annotate_task, extract_annotations, versioned_output_path, DEFAULT_MODEL, target_ids_from_csv, run_annotation_eval
from Benchmarks.Annotations.run_annotations import DEFAULT_NUM_SAMPLES


def get_annotated_sample_ids(annotation_csv_path: str) -> Set[str]:
    """Extract the set of sample IDs that have been annotated from the CSV file."""
    annotated_ids = set()

    if not os.path.exists(annotation_csv_path):
        return annotated_ids

    with open(annotation_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sample_id = row['sample id']  # StepGame uses string IDs
            annotated_ids.add(sample_id)

    return annotated_ids


def record_to_sample(record: Dict[str, Any]) -> Sample:
    input = "story: \n"
    for fact in record["story"]:
        input += f"- {fact}\n"
    input += f"question: {record['question']}"
    target = record["label"]

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{input}{target}".encode()).hexdigest()[:8]

    return Sample(
        input=input,
        target=target,
        id=f"stepgame_{content_hash}",
    )


@task
def stepgame_task() -> Task:
    dataset = hf_dataset("ZhengyanShi/StepGame",
        split="test",
        sample_fields=record_to_sample,
    )

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "stepgame_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )


def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite", model: str = DEFAULT_MODEL, timestamp: str = "", sample_fraction: float = 1.0):
    dataset = hf_dataset("ZhengyanShi/StepGame",
        split="test",
        sample_fields=record_to_sample,
    )
    output_path = os.path.join(Path(__file__).parent, "stepgame_annotations.csv")

    target_ids = target_ids_from_csv(
        output_path,
        num_samples=num_samples if sample_fraction < 1.0 else None,
    )

    if mode == "append":
        resume_path = versioned_output_path(output_path, model, timestamp) if timestamp else output_path
        if os.path.exists(resume_path):
            already_done = {str(sid) for sid in get_annotated_sample_ids(resume_path)}
            target_ids = target_ids - already_done
        if not target_ids:
            print(f"All target samples already annotated for this task.")
            return

    dataset = dataset.filter(lambda sample: str(sample.id) in target_ids)

    annotation_task = annotate_task(dataset)
    log = run_annotation_eval(annotation_task, model=model)
    if timestamp:
        output_path = versioned_output_path(output_path, model, timestamp)
    extract_annotations(log[0], output_path, "overwrite" if timestamp else mode)

if __name__ == "__main__":
    annotate()


