import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice
from inspect_ai.solver import Choices, multiple_choice
from inspect_ai._util.answer import answer_character, answer_index

from Benchmarks.Annotations.annotate_tasks import annotate_task, extract_annotations, versioned_output_path, DEFAULT_MODEL, target_ids_from_csv, run_annotation_eval
from Benchmarks.Annotations.run_annotations import DEFAULT_NUM_SAMPLES

SINGLE_ANSWER_TEMPLATE_COT = r"""
Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of {letters}. Think step by step before answering.

{question}

{choices}
""".strip()


def get_annotated_sample_ids(annotation_csv_path: str) -> Set[int]:
    """Extract the set of sample IDs that have been annotated from the CSV file."""
    annotated_ids = set()

    if not os.path.exists(annotation_csv_path):
        return annotated_ids

    with open(annotation_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sample_id = int(row['sample id'])
            annotated_ids.add(sample_id)

    return annotated_ids


def record_to_sample(record: Dict[str, Any], dataset_path: str, id : int = 0) -> list[Sample]:

    target_scores = list(record["target_scores"].values())
    correct_index = target_scores.index(1)
    target = chr(ord('A') + correct_index)
    sample = Sample(
        id=id,
        input= f"Identify statements that are essential to answer a question: {record['input']}",
        choices=record["target_scores"].keys(),
        target=target,
        )

    return [sample]


def custom_loader(dataset_path: str) -> Dataset:
    
    json_data = json.load(open(dataset_path, 'r'))
    samples = []
    # Iterate through the samples in the record
    for item in json_data["examples"]:
        new_samples = record_to_sample(item, dataset_path, id=len(samples))
        samples.extend(new_samples)
    return MemoryDataset(samples=samples, name="evaluating_information_essentiality", location=dataset_path, shuffled=False)


@task
def evaluating_information_essentiality_task(
    dataset_path: str | None = None,
    ) -> Task:
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "evaluating_information_essentiality.json")
    dataset = custom_loader(dataset_path=dataset_path)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "evaluating_information_essentiality_annotations.csv")
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

def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite", model: str = DEFAULT_MODEL, timestamp: str = "", sample_fraction: float = 1.0):
    dataset_path = os.path.join(Path(__file__).parent, "evaluating_information_essentiality.json")
    output_path = os.path.join(Path(__file__).parent, "evaluating_information_essentiality_annotations.csv")
    dataset = custom_loader(dataset_path=dataset_path)
    dataset = convert_input_to_string(dataset)

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







