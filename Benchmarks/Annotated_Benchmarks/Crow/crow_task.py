import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, generate, multiple_choice
from inspect_ai._util.answer import answer_character, answer_index

from Benchmarks.Annotations.annotate_tasks import annotate_task, extract_annotations, versioned_output_path, DEFAULT_MODEL, target_ids_from_csv, run_annotation_eval
from Benchmarks.Annotations.run_annotations import DEFAULT_NUM_SAMPLES

DIALOGUE_INSTRUCTIONS = "You are a helpful assistant for dialogue understanding. Given the following dialogue between person A and B, answer whether the given response can plausibly follow this dialogue. Answer only 'Yes' or 'No'."

INTENT_INSTRUCTIONS = "You are a helpful assistant for intent classification. Given a news headline and a news writer's intent, answer whether the intent is correct for the headline. Answer only Yes or No."

SAFETY_INSTRUCTIONS = "You are a helpful assistant for safety detection. Given a real-life scenario and an action, answer whether the action is safe to do in this scenario. Answer only Yes or No."

STANCE_INSTRUCTIONS = "You are a helpful assistant for stance classification. Given a belief and an argument, answer whether the argument supports the belief. Answer only Yes or No."

SUMMARIZATION_INSTRUCTIONS = "You are a helpful assistant for dialogue summarization. Given the following dialogue between #Person1# and #Person2#, answer whether the given summary correctly summarizes the dialogue. Answer only 'Yes' or 'No'."


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

    if record["task"] == "dialogue":
        input = DIALOGUE_INSTRUCTIONS + "\nDialogue:\n"
        for response in record["dialogue"]:
            input += f"{response} \n"
        input += f"Response:\n{record['response']}\n"

        target = ""

    elif record["task"] == "intent":
        input = record["headline"]
        target = record["intent"]

    elif record["task"] == "safety":
        input = record["scenario"]
        target = record["action"]

    elif record["task"] == "stance":
        input = record["belief"]
        target = record["argument"]

    elif record["task"] == "summarization":
        input = ""
        for response in record["dialogue"]:
            input += f"{response} \n"
        target = record["summary"]

    else:
        raise ValueError(f"Unknown task type: {record['task']}")

    sample = Sample(
        id=id,
        input= input,
        target=target,
        )

    return [sample]


def custom_loader(dataset_dir: str) -> Dataset:
    
    samples = []
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]

    for json_file in json_files:
        json_data = json.load(open(os.path.join(dataset_dir, json_file), 'r'))
        for item in json_data:
            new_samples = record_to_sample(item, dataset_dir, id=len(samples))
            samples.extend(new_samples)

    return MemoryDataset(samples=samples, name="Crow", location=dataset_dir, shuffled=False)


@task
def crow_task(
    dataset_dir: str | None = None,
    ) -> Task:
    if dataset_dir is None:
        dataset_dir = os.path.join(Path(__file__).parent)
    dataset = custom_loader(dataset_dir=dataset_dir)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "crow_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=choice(),
                solver=generate(),
        )




def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite", model: str = DEFAULT_MODEL, timestamp: str = "", sample_fraction: float = 1.0):
    dataset_dir = os.path.join(Path(__file__).parent)
    output_path = os.path.join(Path(__file__).parent, "crow_annotations.csv")
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








