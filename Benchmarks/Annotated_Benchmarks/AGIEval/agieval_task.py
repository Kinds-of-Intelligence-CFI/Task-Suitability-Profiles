import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character, answer_index

from Annotations.annotate_tasks import annotate_task, extract_annotations
from Annotations.run_annotations import DEFAULT_NUM_SAMPLES

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


def record_to_sample(record: Dict[str, Any], dataset_path: str, mcq: bool, id : int = 0) -> list[Sample]:

    if record.get("passage") is not None:
        input_text = record["passage"]
    else:
        input_text = record["question"]

    if record.get("answer") is not None:
        if mcq:
            return []  # skip this sample if this is an mcq task
        target = record["answer"]
        choices = None
    else:
        if not mcq:
            return []  # skip this sample if this is a freeform task
        target = record["label"]
        choices = record["options"]

    sample = Sample(
        id=id,
        input= input_text,
        choices=choices,
        target=target,
        )

    return [sample]


def custom_loader(dataset_dir: str, mcq: bool) -> Dataset:
    
    samples = []
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jsonl')]

    for json_file in json_files:
        json_data = open(os.path.join(dataset_dir, json_file), 'r')
        sample_data = [json.loads(line) for line in json_data]
        for item in sample_data:
            new_samples = record_to_sample(item, dataset_dir, id=len(samples), mcq=mcq)
            samples.extend(new_samples)

    return MemoryDataset(samples=samples, name="AGIEval", location=dataset_dir, shuffled=False)


@task
def agieval_mcq_task(
    dataset_dir: str | None = None,
    ) -> Task:
    if dataset_dir is None:
        dataset_dir = os.path.join(Path(__file__).parent, "v1_1")
    dataset = custom_loader(dataset_dir=dataset_dir, mcq=True)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "agieval_mcq_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=choice(),
                solver=multiple_choice(cot=True),
        )

@task
def agieval_freeform_task(
    dataset_dir: str | None = None,
    ) -> Task:
    if dataset_dir is None:
        dataset_dir = os.path.join(Path(__file__).parent, "v1_1")
    dataset = custom_loader(dataset_dir=dataset_dir, mcq=False)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "agieval_freeform_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
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
        if sample.choices is not None:
            new_input = prompt(question=sample.input, choices=Choices(sample.choices), template=SINGLE_ANSWER_TEMPLATE_COT)
        else:
            new_input = sample.input
        sample.input = new_input

    return dataset

def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite"):
    dataset_dir = os.path.join(Path(__file__).parent, "v1_1")
    output_path_mcq = os.path.join(Path(__file__).parent, "agieval_mcq_annotations.csv")
    dataset_mcq = custom_loader(dataset_dir=dataset_dir, mcq=True)
    dataset_mcq = convert_input_to_string(dataset_mcq)

    # Shuffle dataset for reproducibility FIRST
    dataset_mcq.shuffle(42)

    if mode == "append":
        already_annotated_ids = get_annotated_sample_ids(output_path_mcq)
        if already_annotated_ids:
            dataset_mcq = dataset_mcq.filter(lambda sample: sample.id not in already_annotated_ids)
        # Calculate remaining samples needed
        remaining_samples = len(dataset_mcq)
        if remaining_samples <= 0:
            print(f"All MCQ samples already annotated. Skipping MCQ annotation.")
        else:
            # Take only what we need
            dataset_mcq = dataset_mcq[:min(num_samples, remaining_samples)]
            annotation_task = annotate_task(dataset_mcq)
            log = eval(annotation_task, model="openai/azure/gpt-4o" )
            extract_annotations(log[0], output_path_mcq, mode)
    else:
        # Overwrite mode - take first num_samples after shuffle
        dataset_mcq = dataset_mcq[:num_samples]
        annotation_task = annotate_task(dataset_mcq)
        log = eval(annotation_task, model="openai/azure/gpt-4o" )
        extract_annotations(log[0], output_path_mcq, mode)


    output_path_freeform = os.path.join(Path(__file__).parent, "agieval_freeform_annotations.csv")
    dataset_freeform = custom_loader(dataset_dir=dataset_dir, mcq=False)
    dataset_freeform = convert_input_to_string(dataset_freeform)

    # Shuffle dataset for reproducibility FIRST
    dataset_freeform.shuffle(42)

    if mode == "append":
        already_annotated_ids_freeform = get_annotated_sample_ids(output_path_freeform)
        if already_annotated_ids_freeform:
            dataset_freeform = dataset_freeform.filter(lambda sample: sample.id not in already_annotated_ids_freeform)
        # Calculate remaining samples needed
        remaining_samples_freeform = len(dataset_freeform)
        if remaining_samples_freeform <= 0:
            print(f"All freeform samples already annotated. Skipping freeform annotation.")
            return
        else:
            # Take only what we need
            dataset_freeform = dataset_freeform[:min(num_samples, remaining_samples_freeform)]
    else:
        # Overwrite mode - take first num_samples after shuffle
        dataset_freeform = dataset_freeform[:num_samples]

    annotation_task = annotate_task(dataset_freeform)
    log = eval(annotation_task, model="openai/azure/gpt-4o" )
    extract_annotations(log[0], output_path_freeform, mode)


if __name__ == "__main__":
    annotate()







