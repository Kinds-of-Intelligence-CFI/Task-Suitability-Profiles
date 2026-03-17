

import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice, match, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, generate, multiple_choice, system_message
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


def record_to_sample(record: Dict[str, Any], prompt:str, id : int = 0) -> list[Sample]:

    input = record["input"]
    target = record["target"]

    sample = Sample(
        id=id,
        input= input,
        target=target,
        metadata={"prompt": prompt}
        )

    return [sample]


def custom_loader(dataset_dir: str, prompt_dir:str) -> Dataset:
    
    samples = []
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]

    task_prompts = {}
    prompt_files = [f for f in os.listdir(prompt_dir) if f.endswith('.txt')]
    for prompt_file in prompt_files:
        prompt_data = open(os.path.join(prompt_dir, prompt_file), 'r').read()
        # next strip out the first 2 lines which are comments
        prompt_data = "\n".join(prompt_data.split("\n")[2:])
        task_name = prompt_file.replace(".txt", "")
        task_prompts[task_name] = prompt_data

    for json_file in json_files:
        assert json_file.replace(".json", "") in task_prompts, f"No prompt found for {json_file}"
        prompt = task_prompts[json_file.replace(".json", "")]
        json_data = json.load(open(os.path.join(dataset_dir, json_file), 'r'))
        for sample in json_data["examples"]:
            new_samples = record_to_sample(sample, prompt, id=len(samples))
            samples.extend(new_samples)


    return MemoryDataset(samples=samples, name="BigBenchHard", location=dataset_dir, shuffled=False)


@task
def bigbenchhard_task(
    dataset_dir: str | None = None,
    prompt_dir: str | None = None,
    ) -> Task:
    if dataset_dir is None:
        dataset_dir = os.path.join(Path(__file__).parent, "bbh")
    if prompt_dir is None:
        prompt_dir = os.path.join(Path(__file__).parent, "cot-prompts")
    dataset = custom_loader(dataset_dir=dataset_dir, prompt_dir=prompt_dir)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "bigbenchhard_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=match(),
                solver=[system_message("{prompt}"), generate()],
        )



def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite"):
    dataset_dir = os.path.join(Path(__file__).parent, "bbh")
    prompt_dir = os.path.join(Path(__file__).parent, "cot-prompts")
    output_path = os.path.join(Path(__file__).parent, "bigbenchhard_annotations.csv")
    dataset = custom_loader(dataset_dir=dataset_dir, prompt_dir=prompt_dir)

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











