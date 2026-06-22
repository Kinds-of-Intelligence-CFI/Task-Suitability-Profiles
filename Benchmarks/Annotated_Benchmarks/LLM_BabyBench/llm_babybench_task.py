import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, Sample, hf_dataset
from inspect_ai.scorer import Score, Scorer, Target, accuracy, choice, includes, match, scorer, stderr
from inspect_ai.solver import Choices, TaskState, basic_agent, generate, multiple_choice
from inspect_ai._util.answer import answer_character

from Benchmarks.Annotations.annotate_tasks import annotate_task, extract_annotations, versioned_output_path, DEFAULT_MODEL, target_ids_from_csv, run_annotation_eval
from Benchmarks.Annotations.run_annotations import DEFAULT_NUM_SAMPLES
from Benchmarks.Annotated_Benchmarks.LLM_BabyBench.decompose import DecomposeEvaluator
from Benchmarks.Annotated_Benchmarks.LLM_BabyBench.plan import PlanEvaluator
from Benchmarks.Annotated_Benchmarks.LLM_BabyBench.register import register_envs
import gymnasium as gym
from gymnasium import Env


def get_annotated_sample_ids(annotation_csv_path: str) -> Set[str]:
    """Extract the set of sample IDs that have been annotated from the CSV file."""
    annotated_ids = set()

    if not os.path.exists(annotation_csv_path):
        return annotated_ids

    with open(annotation_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sample_id = row['sample id']  # LLM_BabyBench uses string IDs
            annotated_ids.add(sample_id)

    return annotated_ids


def predict_record_to_sample(record: Dict[str, Any]) -> Sample:
    input = f"{record['env_description']} What state would the agent be in if it took the following actions?:\n initial_state: {record['initial_state']}\n{record['action_sequence']} \n give your answer in the form ((x, y), d) where d is the direction the agent is facing (east=0, south=1, west=2, north=3)."
    target=record["target_state"]

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{input}{target}{record['level_name']}".encode()).hexdigest()[:8]

    sample = Sample(input=input,
                    target=target,
                    metadata=record,
                    id=f"babybench_predict_{content_hash}",
                    )
    return sample

@task
def predict_task() -> Task:
    dataset = hf_dataset("salem-mbzuai/LLM-BabyBench",
                         name="predict",
                         split="train",
                         sample_fields=predict_record_to_sample
                         )

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "llm_babybench_predict_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                solver=basic_agent(),
                scorer=match(),
                )

def plan_record_to_sample(record: Dict[str, Any]) -> Sample:
    input = f"{record['env_description']} What actions should be taken to reach the target sub goal?:\n initial_state ((x, y), d) where d is the direction the agent is facing (east=0, south=1, west=2, north=3): {record['initial_state']}\n target sub goal: {record['target_subgoal']} \n give your answer as a comma seperated list of actions. The names of possible actions are again: left, right, forward, pickup, drop, toggle."
    target=record["expert_action_sequence"]

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{input}{target}{record['level_name']}".encode()).hexdigest()[:8]

    sample = Sample(input=input,
                    target=target,
                    metadata=record,
                    id=f"babybench_plan_{content_hash}",
                    )
    return sample

@scorer(metrics=[accuracy(), stderr()])
def plan_scorer()-> Scorer:
    """This is a task specific scorer that will check if the action sequence ends up in the correct location."""
    async def score(state: TaskState, target: Target) -> Score:
        # first create an environment with the correct setup for the sample
        env_name = state.metadata["level_name"]
        seed = state.metadata["seed"]

        # then apply the evaluate function from the original paper
        evaluator = PlanEvaluator()
        try:
            result = evaluator.evaluate(env_name=env_name, seed=seed, optimal_action_seq=target.text, llm_action_seq=state.output.completion)
        except ValueError as e:
            result = {
                "CR": 0,
                "PR": 0,
                "ACI" : 0,
            }
        # finally convert that to a score
        return Score(value=result["CR"])
    return score

@task
def plan_task() -> Task:
    dataset = hf_dataset("salem-mbzuai/LLM-BabyBench",
                         name="plan",
                         split="train",
                         sample_fields=plan_record_to_sample
                         )

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "llm_babybench_plan_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                solver=basic_agent(),
                scorer=[plan_scorer()],
                )

def decompose_record_to_sample(record: Dict[str, Any]) -> Sample:
    input = f"{record['env_description']} What subgoals should be taken in order to achive the mission?:\n initial_state ((x, y), d) where d is the direction the agent is facing (east=0, south=1, west=2, north=3): {record['initial_state']}\n mission: {record['mission']} \n sub goals should be in the format and are executed in the order you give: "
    input += """- OpenSubgoal: If the agent is next to a closed door, this subgoal will open it. \n\n
- CloseSubgoal: If the agent is next to an open door, this subgoal will close it. \n\n
- DropSubgoal: If the agent is carrying an object, this subgoal drops it. \n\n
- PickupSubgoal: If the agent is not carrying any object and is next to an object, this subgoal picks it up. \n\n
- GoNextToSubgoal, (x,y): If there is a clear, without any blocker, path between the agent and the cell of coordinates (x,y), this subgoal makes the agent go next to this cell."""
    input += "\ngive your answer list of subgoals each on a new line. for example:\n GoNextToSubgoal, (3, 5)\nGoNextToSubgoal, (2, 5)\nGoNextToSubgoal, (1, 5)\nGoNextToSubgoal, (1, 6)\nPickupSubgoal"

    # Create a unique identifier based on content hash since no index is available
    import hashlib
    content_hash = hashlib.md5(f"{input}{record['level_name']}".encode()).hexdigest()[:8]

    sample = Sample(input=input,
                    metadata=record,
                    id=f"babybench_decompose_{content_hash}",
                    )
    return sample


def make_env(env_name: str, seed: int) -> Env:
    """
    Create an environment from config file specifications.
    """
    register_envs()
    env = gym.make(env_name, tile_size=32, render_mode='rgb_array')
    env.reset(seed=seed)
    return env

@scorer(metrics=[accuracy(), stderr()])
def decompose_scorer()-> Scorer:
    """This is a task specific scorer that will check if the action sequence ends up in the correct location."""
    async def score(state: TaskState, target: Target) -> Score:
        # first create an environment with the correct setup for the sample
        env_name = state.metadata["level_name"]
        seed = state.metadata["seed"]
        env = make_env(env_name, seed)

        # then apply the evaluate function from the original paper
        evaluator = DecomposeEvaluator()
        try:
            result = evaluator.evaluate(env, state.output.completion)
        except ValueError as e:
            result = {
                "CR": 0,
                "PR": 0,
                "ACI" : 0,
            }
        # finally convert that to a score
        return Score(value=result["CR"])
    return score

@task
def decompose_task() -> Task:
    dataset = hf_dataset("salem-mbzuai/LLM-BabyBench",
                         name="decompose",
                         split="train",
                         sample_fields=decompose_record_to_sample
                         )

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "llm_babybench_decompose_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                solver=basic_agent(),
                scorer=decompose_scorer(),
                )


def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite", model: str = DEFAULT_MODEL, timestamp: str = "", sample_fraction: float = 1.0):
    # Annotate predict task
    output_path_predict = os.path.join(Path(__file__).parent, "llm_babybench_predict_annotations.csv")
    dataset_predict = hf_dataset("salem-mbzuai/LLM-BabyBench",
                                 name="predict",
                                 split="train",
                                 sample_fields=predict_record_to_sample
                                 )

    target_ids_predict = target_ids_from_csv(
        output_path_predict,
        num_samples=num_samples if sample_fraction < 1.0 else None,
    )

    predict_out = versioned_output_path(output_path_predict, model, timestamp) if timestamp else output_path_predict
    if mode == "append" and os.path.exists(predict_out):
        already_done = {str(sid) for sid in get_annotated_sample_ids(predict_out)}
        target_ids_predict = target_ids_predict - already_done

    if not target_ids_predict:
        print("All predict samples already annotated. Skipping predict annotation.")
    else:
        dataset_predict = dataset_predict.filter(lambda sample: str(sample.id) in target_ids_predict)
        annotation_task = annotate_task(dataset_predict)
        log = run_annotation_eval(annotation_task, model=model)
        extract_annotations(log[0], predict_out, "overwrite" if timestamp else mode)

    # Annotate plan task
    output_path_plan = os.path.join(Path(__file__).parent, "llm_babybench_plan_annotations.csv")
    dataset_plan = hf_dataset("salem-mbzuai/LLM-BabyBench",
                              name="plan",
                              split="train",
                              sample_fields=plan_record_to_sample
                              )

    target_ids_plan = target_ids_from_csv(
        output_path_plan,
        num_samples=num_samples if sample_fraction < 1.0 else None,
    )

    plan_out = versioned_output_path(output_path_plan, model, timestamp) if timestamp else output_path_plan
    if mode == "append" and os.path.exists(plan_out):
        already_done = {str(sid) for sid in get_annotated_sample_ids(plan_out)}
        target_ids_plan = target_ids_plan - already_done

    if not target_ids_plan:
        print("All plan samples already annotated. Skipping plan annotation.")
    else:
        dataset_plan = dataset_plan.filter(lambda sample: str(sample.id) in target_ids_plan)
        annotation_task = annotate_task(dataset_plan)
        log = run_annotation_eval(annotation_task, model=model)
        extract_annotations(log[0], plan_out, "overwrite" if timestamp else mode)

    # Annotate decompose task
    output_path_decompose = os.path.join(Path(__file__).parent, "llm_babybench_decompose_annotations.csv")
    dataset_decompose = hf_dataset("salem-mbzuai/LLM-BabyBench",
                                   name="decompose",
                                   split="train",
                                   sample_fields=decompose_record_to_sample
                                   )

    target_ids_decompose = target_ids_from_csv(
        output_path_decompose,
        num_samples=num_samples if sample_fraction < 1.0 else None,
    )

    decompose_out = versioned_output_path(output_path_decompose, model, timestamp) if timestamp else output_path_decompose
    if mode == "append" and os.path.exists(decompose_out):
        already_done = {str(sid) for sid in get_annotated_sample_ids(decompose_out)}
        target_ids_decompose = target_ids_decompose - already_done

    if not target_ids_decompose:
        print("All decompose samples already annotated. Skipping decompose annotation.")
    else:
        dataset_decompose = dataset_decompose.filter(lambda sample: str(sample.id) in target_ids_decompose)
        annotation_task = annotate_task(dataset_decompose)
        log = run_annotation_eval(annotation_task, model=model)
        extract_annotations(log[0], decompose_out, "overwrite" if timestamp else mode)


if __name__ == "__main__":
    annotate()
