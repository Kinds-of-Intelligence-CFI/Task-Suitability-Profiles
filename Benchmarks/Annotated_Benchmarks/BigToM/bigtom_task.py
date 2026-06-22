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

from Benchmarks.Annotations.annotate_tasks import annotate_task, extract_annotations, versioned_output_path, DEFAULT_MODEL, target_ids_from_csv, run_annotation_eval
from Benchmarks.Annotations.run_annotations import DEFAULT_NUM_SAMPLES


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


def record_to_sample(record: list[str], dataset_path: str, id : int = 0) -> list[Sample]:
    input = record[0]
    question = record[1]
    target = record[2]

    sample = Sample(
        id=id,
        input= f"context: {input} \n question: {question}",
        target=target,
        )

    return [sample]


def custom_loader(dataset_dir: str) -> Dataset:
    
    samples = []

    for foldername in os.listdir(dataset_dir):
        if not os.path.isdir(os.path.join(dataset_dir, foldername)):
            continue
        folder_path = os.path.join(dataset_dir, foldername)
        for filename in os.listdir(folder_path):
            csv_file = open(os.path.join(folder_path, filename), "r")
            for line in csv_file:
                record = line.split(";")
                new_samples = record_to_sample(record=record, dataset_path=dataset_dir, id=len(samples))
                samples.extend(new_samples)
    return MemoryDataset(samples=samples, name="BigToM", location=dataset_dir, shuffled=False)


@task
def bigtom_task(
    dataset_dir: str | None = None,
    ) -> Task:
    if dataset_dir is None:
        dataset_dir = os.path.join(Path(__file__).parent, "conditions")
    dataset = custom_loader(dataset_dir=dataset_dir)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "bigtom_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )



def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite", model: str = DEFAULT_MODEL, timestamp: str = "", sample_fraction: float = 1.0):
    dataset_dir = os.path.join(Path(__file__).parent, "conditions")
    output_path = os.path.join(Path(__file__).parent, "bigtom_annotations.csv")
    dataset = custom_loader(dataset_dir=dataset_dir)

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







