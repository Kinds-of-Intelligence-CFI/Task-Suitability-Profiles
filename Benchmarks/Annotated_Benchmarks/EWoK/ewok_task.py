import os
import csv
from pathlib import Path
from typing import Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import choice, includes
from inspect_ai.solver import Choices, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character

from Annotations.annotate_tasks import annotate_task, extract_annotations
from Annotations.run_annotations import DEFAULT_NUM_SAMPLES

current_hashes = []

INPUT_TEMPLATE = """
# INSTRUCTIONS

In this study, you will see multiple examples. In each example, you will be given two contexts and a scenario. Your task is to read the two contexts and the subsequent scenario, and pick the context that makes more sense considering the scenario that follows. The contexts will be numbered "1" or "2". You must answer using "1" or "2" in your response.


# TRIAL EXAMPLE

## Contexts
1. "The bag is full of blocks."
2. "The bag is full of balls."

## Scenario
"I drew a ball from the bag."

## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
2


# TRIAL EXAMPLE

## Contexts
1. "The boy likes cookies."
2. "The boy does not like cookies."

## Scenario
"The boy chose to eat a cookie."

## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
1


# TEST EXAMPLE

## Contexts
1. "{context_1}"
2. "{context_2}"

## Scenario
"{scenario}"

## Task
Which context makes more sense given the scenario? Please answer using either "1" or "2".

## Response
"""


def get_annotated_sample_ids(annotation_csv_path: str) -> Set[str]:
    """Extract the set of sample IDs that have been annotated from the CSV file."""
    annotated_ids = set()

    if not os.path.exists(annotation_csv_path):
        return annotated_ids

    with open(annotation_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sample_id = row['sample id']  # EWoK uses string IDs
            annotated_ids.add(sample_id)

    return annotated_ids


def record_to_sample(record) -> list[Sample]:
    context_1 = record["Context1"]
    context_2 = record["Context2"]
    scenario_1 = record["Target1"]
    scenario_2 = record["Target2"]

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{context_1}{context_2}{scenario_1}{scenario_2}".encode()).hexdigest()[:8]

    if content_hash in current_hashes:
        return []
    else:
        current_hashes.append(content_hash)

    sample_1 = Sample(input= INPUT_TEMPLATE.format(context_1=context_1, context_2=context_2, scenario=scenario_1),
                      target="1",
                      id=f"ewok_{content_hash}_1")
    sample_2 = Sample(input= INPUT_TEMPLATE.format(context_1=context_1, context_2=context_2, scenario=scenario_2),
                      target="2",
                      id=f"ewok_{content_hash}_2")
    return [sample_1, sample_2]

@task
def ewok_task() -> Task:
    dataset = hf_dataset("ewok-core/ewok-core-1.0",
        split="test",
        sample_fields=record_to_sample,
    )

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "ewok_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=includes(),
                solver=basic_agent(),
        )

def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite"):
    dataset = hf_dataset("ewok-core/ewok-core-1.0",
        split="test",
        sample_fields=record_to_sample,
    )
    output_path = os.path.join(Path(__file__).parent, "ewok_annotations.csv")

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


