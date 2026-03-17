import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import Choices, multiple_choice
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
            sample_id = row['sample id']  # MetaMedQA uses string IDs
            annotated_ids.add(sample_id)

    return annotated_ids


def record_to_sample(record: Dict[str, Any]) -> Sample:
    input = record["question"]
    target = record["answer_idx"]
    choices = record["options"].values()

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{input}{''.join(choices)}{str(target)}".encode()).hexdigest()[:8]

    return Sample(
        input=input,
        target=target,
        choices=choices,
        id=f"metamedqa_{content_hash}",
    )


@task
def metamedqa_task() -> Task:
    dataset = hf_dataset("maximegmd/MetaMedQA",
        split="test",
        sample_fields=record_to_sample,
    )

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "metamedqa_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=choice(),
                solver=multiple_choice(cot=True),
        )

def answer_options(choices: Choices) -> str:
    r"""
    Returns the `choices` formatted as a multiple choice question, e.g.:

    ["choice 1", "choice 2", "choice 3"] ->
        "A) choice 1\nB) choice 2\nC) choice 3"
    """
    indexes = list(range(len(choices)))

    return "\n".join(
        [f"{answer_character(i)}) {choices[j].value}" for i, j in enumerate(indexes)]
    )

def prompt(question: str, choices: Choices, template: str) -> str:
    choices_text = answer_options(choices)
    letters = ",".join(answer_character(i) for i in range(len(choices)))

    return template.format(
        choices=choices_text,
        letters=letters,
        question=question,
    )


def convert_input_to_string(dataset: Dataset) -> Dataset:
    """takes a dataset and converts the inputs from messages to strings for annotation only"""
    for sample in dataset:
        new_input = prompt(question=sample.input, choices=Choices(sample.choices), template=SINGLE_ANSWER_TEMPLATE_COT)
        sample.input = new_input

    return dataset

def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite"):
    dataset = hf_dataset("maximegmd/MetaMedQA",
        split="test",
        sample_fields=record_to_sample,
    )
    dataset = convert_input_to_string(dataset)
    output_path = os.path.join(Path(__file__).parent, "metamedqa_annotations.csv")

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
    log = eval(annotation_task, model="openai/azure/gpt-4o" )
    extract_annotations(log[0], output_path, mode)

if __name__ == "__main__":
    annotate()


