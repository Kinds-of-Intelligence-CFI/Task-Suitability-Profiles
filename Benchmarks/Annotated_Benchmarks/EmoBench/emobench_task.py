import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import choice, model_graded_qa
from inspect_ai.solver import Choices, basic_agent, multiple_choice, system_message
from inspect_ai._util.answer import answer_character, answer_index

from Benchmarks.Annotations.annotate_tasks import annotate_task, extract_annotations, versioned_output_path, DEFAULT_MODEL, target_ids_from_csv, run_annotation_eval
from Benchmarks.Annotations.run_annotations import DEFAULT_NUM_SAMPLES

SYSTEM_PROMPT = "In this task, you are presented with a scenario, a question, and multiple choices. Carefully analyze the scenario and take the perspective of the individual involved.Then, select the option that best reflects their perspective or emotional response."

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
            sample_id = row['sample id']  # EmoBench uses string IDs
            annotated_ids.add(sample_id)

    return annotated_ids


def record_to_sample(record: Dict[str, Any], json_file: str, id : int = 0) -> list[Sample]:

    samples = []
    if json_file == "EA.jsonl":
        context = record["scenario"]
        subject = record["subject"]
        choices = record["choices"]
        correct_answer_string = record["label"]
        # now find the index of the correct answer
        correct_answer_index = record["choices"].index(correct_answer_string)
        answer = answer_character(correct_answer_index)
        q_type = record["question type"]
        input = f"""
    ## Scenario
    {context}
    
    ## Question 
    In this scenario, what is the most effective {q_type} for {subject}?
    
    ## Choices"""
        sample = Sample(
            id = f"ea_{record['qid']}_{record['language']}",
            input=input,
            choices=choices,
            target=answer,
        )
        samples.append(sample)
    elif json_file == "EU.jsonl":
        context = record["scenario"]
        subject = record["subject"]
        emo_choices = record["emotion_choices"]
        correct_emo_answer_string = record["emotion_label"]
        # now find the index of the correct answer
        correct_emo_answer_index = record["emotion_choices"].index(correct_emo_answer_string)
        emo_answer = answer_character(correct_emo_answer_index)
        input = f"""
    ## Scenario
    {context}
    
    ## Question
    What emotion(s) would {subject} ultimately feel in this situation?
    
    ## Choices for Question"""
        emo_sample = Sample(
            id = f"eu_emo_{record['qid']}_{record['language']}",
            input=input,
            choices=emo_choices,
            target=emo_answer,
        )
        samples.append(emo_sample)

        cause_choices = record["cause_choices"]
        correct_cause_answer_string = record["cause_label"]
        # now find the index of the correct answer
        correct_cause_answer_index = record["cause_choices"].index(correct_cause_answer_string)
        cause_answer = answer_character(correct_cause_answer_index)
        input = f"""
    ## Scenario
    {context}
    
    ## Question
    Why would {subject} feel {correct_emo_answer_string} situation?

    ## Choices for Question"""
        cause_sample = Sample(
            id = f"eu_cause_{record['qid']}_{record['language']}",
            input=input,
            choices=cause_choices,
            target=cause_answer,
        )
        samples.append(cause_sample)
    else:
        raise ValueError(f"Unknown json file: {json_file}")


    return samples


def custom_loader(dataset_dir: str) -> Dataset:
    
    samples = []
    json_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jsonl')]

    for json_file in json_files:
        json_data = open(os.path.join(dataset_dir, json_file), 'r')
        sample_data = [json.loads(line) for line in json_data]
        for item in sample_data:
            new_samples = record_to_sample(item, json_file, id=len(samples))
            samples.extend(new_samples)

    return MemoryDataset(samples=samples, name="EmoBench", location=dataset_dir, shuffled=False)


@task
def emobench_task(
    dataset_dir: str | None = None,
    ) -> Task:
    if dataset_dir is None:
        dataset_dir = os.path.join(Path(__file__).parent)
    dataset = custom_loader(dataset_dir=dataset_dir)

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "emobench_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=choice(),
                solver=[system_message(SYSTEM_PROMPT),
                        multiple_choice(cot=True)],
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

def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite", model: str = DEFAULT_MODEL, timestamp: str = "", sample_fraction: float = 1.0):
    dataset_dir = os.path.join(Path(__file__).parent)
    output_path = os.path.join(Path(__file__).parent, "emobench_annotations.csv")
    dataset = custom_loader(dataset_dir=dataset_dir)
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







