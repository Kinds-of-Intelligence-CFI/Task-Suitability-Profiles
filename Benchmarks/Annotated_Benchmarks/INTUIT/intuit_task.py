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


def record_to_sample(record: Dict[str, Any], dataset_path: str, id : int = 0) -> Sample:
    vignette = record['vignette'].strip()
    
    # Split the vignette into question and choices
    lines = vignette.split('\n')
    
    # Find where the numbered choices start
    question_lines = []
    choices = []
    
    for line in lines:
        line = line.strip()
        if line.startswith(('1.', '2.', '3.', '4.')):
            # Extract the choice text (remove number prefix)
            choice_text = line[2:].strip()
            choices.append(choice_text)
        elif line:  # Non-empty line that's not a choice
            question_lines.append(line)
    
    # Join the question lines back together
    question = '\n'.join(question_lines).strip()
    
    correct_answer_num = int(record['correct_answer_number'])
    target = answer_character(correct_answer_num - 1)
    
    return Sample(
        input=question,
        target=target,
        choices=choices,
        id=id
    )


def custom_loader(dataset_path: str) -> Dataset:
    samples = []
    
    with open(dataset_path, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for idx, record in enumerate(reader):
            sample = record_to_sample(record, dataset_path, idx)
            samples.append(sample)
    
    return MemoryDataset(samples=samples, name="intuit", location=dataset_path, shuffled=False)


@task
def intuit_task(
    dataset_path: str | None = None,
    ) -> Task:
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "battery_for_ai_clean.csv")
    dataset = custom_loader(dataset_path=dataset_path)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "intuit_annotations.csv")
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
    dataset_path = os.path.join(Path(__file__).parent, "battery_for_ai_clean.csv")
    output_path = os.path.join(Path(__file__).parent, "intuit_annotations.csv")
    dataset = custom_loader(dataset_path=dataset_path)
    dataset = convert_input_to_string(dataset)

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







