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
    samples = []
    input = record["full_context"]
    fact_question = record["factQA"]
    question_text = fact_question["question"]
    correct_answer = fact_question["correct_answer"]
    wrong_answer = fact_question["wrong_answer"]

    sample = Sample(
        id=id,
        input= f"context: {input} \n question: {question_text}",
        target=correct_answer,
        metadata={"question_type": "fact"},
        )
    samples.append(sample)
    id += 1

    for belief_question in record["beliefQAs"]:
        question = belief_question["question"]
        correct_answer = belief_question["correct_answer"]
        wrong_answer = belief_question["wrong_answer"]
        
        sample = Sample(
            id=id,
            input= f"context: {input} \n question: {question}",
            target=correct_answer,
            metadata={"question_type": "belief"},
            )
        samples.append(sample)
        id += 1

    info_question = record["infoAccessibilityQA_list"]
    question = info_question["question"]
    correct_answer = info_question["correct_answer"]
    wrong_answer = info_question["wrong_answer"]
    
    sample = Sample(
        id=id,
        input= f"context: {input} {question_text} \n question: {question}",
        target=correct_answer,
        metadata={"question_type": "info_accessibility"},
        )
    samples.append(sample)
    id += 1

    answer_question = record["answerabilityQA_list"]
    question = answer_question["question"]
    correct_answer = answer_question["correct_answer"]
    wrong_answer = answer_question["wrong_answer"]
    
    sample = Sample(
        id=id,
        input= f"context: {input} {question_text} \n question: {question}",
        target=correct_answer,
        metadata={"question_type": "answerability"},
        )
    samples.append(sample)
    id += 1

    return samples


def custom_loader(dataset_path: str) -> Dataset:
    
    json_data = json.load(open(dataset_path, 'r'))
    samples = []
    # Iterate through the samples in the record
    for item in json_data:
        new_samples = record_to_sample(item, dataset_path, id=len(samples))
        samples.extend(new_samples)
    return MemoryDataset(samples=samples, name="Fantom", location=dataset_path, shuffled=False)


@task
def fantom_task(
    dataset_path: str | None = None,
    ) -> Task:
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "fantom_v1.json")
    dataset = custom_loader(dataset_path=dataset_path)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "fantom_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=model_graded_qa(),
                solver=basic_agent(),
        )



def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite"):
    dataset_path = os.path.join(Path(__file__).parent, "fantom_v1.json")
    output_path = os.path.join(Path(__file__).parent, "fantom_annotations.csv")
    dataset = custom_loader(dataset_path=dataset_path)

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







