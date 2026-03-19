



# need to make a stateful tool for the agent to take actions
import os
import csv
from pathlib import Path
from typing import Any, Optional, Set
import numpy as np
from inspect_ai import Task, task, eval
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import Score, Scorer, Target, accuracy, scorer, stderr, stderr
from inspect_ai.solver import Generate, Solver, TaskState, basic_agent, solver, use_tools
from inspect_ai.tool import Tool, tool
from inspect_ai.util import StoreModel, message_limit, store_as
from pydantic import Field

from Benchmarks.Annotations.annotate_tasks import annotate_task, extract_annotations, versioned_output_path, DEFAULT_MODEL
from Benchmarks.Annotations.run_annotations import DEFAULT_NUM_SAMPLES
from Benchmarks.Annotated_Benchmarks.Text_Navigation.maze import maze_game


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


class MazeStateModel(StoreModel):
    maze : Optional[maze_game] = None

@tool
def step_tool(state: TaskState, instance: str | None = None) -> Tool:
    """A tool for taking actions in the text maze and maintaining the state."""
    async def execute(action: str) -> str:
        """Take a step in the maze.

        This tool takes a step in the environment. The set of possible actions are:
        north,
        east,
        south,
        west,
        none,   # no action, just returns the state of the environment

        Args:
           action: A string containing the action you wish to take.

        Returns:
           Answer to research prompt or question.
        """
        # get the state of the current environment 
        maze_state = store_as(MazeStateModel, instance=instance)
        if maze_state.maze is None:
            np.random.seed(state.sample_id)
            maze_state.maze = maze_game(map_width=15)

        # parse the action
        action_lower = action.lower()
        if action_lower == "none":
            response = maze_state.maze.pretty_state()
        else:
            response = maze_state.maze.step(action_lower)

        if maze_state.maze.game_over:
            state.completed = True

        return response
    return execute

@solver
def add_step_tool() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state.tools.append(step_tool(state=state))
        return state
    
    return solve


@scorer(metrics=[accuracy(), stderr()])
def message_limit_scorer() -> Scorer:
    """A simple scorer that marks the sample as correct if it ends before the message limit is reached."""
    async def score(state: TaskState, target: Target) -> Score:
        if not state.message_limit:
            value = 0.0
        else:
            value = 1.0 if len(state.messages) < state.message_limit else 0.0
    
        return Score(value=value)
    return score



@task
def text_navigation_task() -> Task:
    input_text = "You enter a wicked maze.  As the (P)layer, you seek the (m)cguffin.  Find it, if you dare.  If you seek help, simply ask for it. Take the none action to get started."

    annotation_csv_path = os.path.join(Path(__file__).parent, "text_navigation_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        samples = [Sample(input=input_text, id=i) for i in sorted(annotated_ids)]
    else:
        samples = [Sample(input=input_text, id=i) for i in range(DEFAULT_NUM_SAMPLES)]

    dataset = MemoryDataset(samples, name="text_navigation")

    return Task(dataset=dataset,
                solver=[
                    add_step_tool(),
                    basic_agent(),
                    ],
                message_limit=30,
                scorer=message_limit_scorer()
                )




def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite", model: str = DEFAULT_MODEL, timestamp: str = ""):
    output_path = os.path.join(Path(__file__).parent, "text_navigation_annotations.csv")

    if mode == "append":
        # Get already annotated sample IDs
        already_annotated_ids = get_annotated_sample_ids(output_path)

        # Calculate remaining samples needed
        remaining_samples = min(num_samples, num_samples - len(already_annotated_ids))

        # Only proceed if there are remaining samples to annotate
        if remaining_samples <= 0:
            print(f"All {len(already_annotated_ids)} samples already annotated. Nothing to do.")
            return

        # Start ID from highest existing ID + 1
        start_id = max(already_annotated_ids) + 1 if already_annotated_ids else 0
        num_samples = remaining_samples
    else:
        start_id = 0

    # make a dataset manually
    samples = []
    for i in range(num_samples):
        input = "You enter a wicked maze.  As the (P)layer, you seek the (m)cguffin.  Find it, if you dare.  If you seek help, simply ask for it. Take the none action to get started."
        np.random.seed(start_id + i)
        maze = maze_game(map_width=15)
        input += f"\n{maze.pretty_state()}"
        samples.append(Sample(input=input, id=start_id + i))

    dataset = MemoryDataset(samples=samples, name="text_navigation")

    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model=model)
    if timestamp:
        output_path = versioned_output_path(output_path, model, timestamp)
    extract_annotations(log[0], output_path, "overwrite" if timestamp else mode)

if __name__ == "__main__":
    annotate()

    


