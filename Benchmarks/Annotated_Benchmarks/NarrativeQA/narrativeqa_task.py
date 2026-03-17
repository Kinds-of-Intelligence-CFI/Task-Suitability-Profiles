import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character

from Annotations.annotate_tasks import annotate_task, extract_annotations
from Annotations.run_annotations import DEFAULT_NUM_SAMPLES

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()


def get_annotated_sample_ids(annotation_csv_path: str) -> Set[str]:
    """Extract the set of sample IDs that have been annotated from the CSV file."""
    annotated_ids = set()

    if not os.path.exists(annotation_csv_path):
        return annotated_ids

    with open(annotation_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sample_id = row['sample id']  # NarrativeQA uses string IDs
            annotated_ids.add(sample_id)

    return annotated_ids


def record_to_sample(record: Dict[str, Any]) -> Sample:
    input = f"context: {record['document']['text']} \n question: {record['question']['text']}"
    target = [answer["text"] for answer in record["answers"]]

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{input}{''.join(target)}".encode()).hexdigest()[:8]

    return Sample(
        input=input,
        target=target,
        id=f"narrativeqa_{content_hash}",
    )


@task
def narrativeqa_task() -> Task:
    dataset = hf_dataset("deepmind/narrativeqa",
        split="test",
        sample_fields=record_to_sample,
    )

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "narrativeqa_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )




def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite"):
    dataset = hf_dataset("deepmind/narrativeqa",
        split="test",
        sample_fields=record_to_sample,
    )
    output_path = os.path.join(Path(__file__).parent, "narrativeqa_annotations.csv")

    # Shuffle dataset for reproducibility FIRST
    dataset.shuffle(42)

    if mode == "append":
        already_annotated_ids = get_annotated_sample_ids(output_path)
        if already_annotated_ids:
            dataset = dataset.filter(lambda sample: sample.id not in already_annotated_ids)
        # Calculate remaining samples needed
        remaining_samples = len(dataset)
        if remaining_samples <= 0:
            print(f"All samples already annotated for this task.")
            return
        # Take only what we need
        dataset = dataset[:min(num_samples, remaining_samples)]
    else:
        # Overwrite mode - take first num_samples after shuffle
        dataset = dataset[:num_samples]

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o", max_connections=2)
    extract_annotations(log[0], output_path, mode)

if __name__ == "__main__":
    annotate()

