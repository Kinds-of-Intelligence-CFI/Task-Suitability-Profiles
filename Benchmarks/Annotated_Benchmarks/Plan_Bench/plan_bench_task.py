import json
import os
import csv
from pathlib import Path
from typing import Any, Dict, Set
from inspect_ai import Task, task, eval
from inspect_ai.dataset import Dataset, FieldSpec, MemoryDataset, Sample, hf_dataset
from inspect_ai.model import ChatMessageUser, ContentText, Model, get_model
from inspect_ai.scorer import Score, Scorer, Target, accuracy, choice, model_graded_qa, scorer, stderr
from inspect_ai.solver import Choices, TaskState, basic_agent, multiple_choice
from inspect_ai._util.answer import answer_character

from Annotations.annotate_tasks import annotate_task, extract_annotations
from Annotations.run_annotations import DEFAULT_NUM_SAMPLES

TRANSLATION_PROMPT = """
Extract the plan from the text below and convert it to a PDDL plan.

Example: 
[RAW TEXT]
To achieve your goal of having both `b6e9q4r60gagvdcn object_0 object_2` and `b6e9q4r60gagvdcn object_1 object_0` true, you can follow this plan:

1. **Action:** `u64y1a9apusmslxb object_0`

   *Preconditions:*
   - `tv30k33pzoulql6w object_0` (true in initial state)
   - `af9tse23ljclsqad object_0` (true in initial state)
   - `cql4o62p1yeke3ok` (true in initial state)

   *Effect:*
   - Adds `a46zhwykn6jvbabk object_0`
   - Deletes `tv30k33pzoulql6w object_0`, `af9tse23ljclsqad object_0`, and `cql4o62p1yeke3ok`

2. **Action:** `j4gv801gnu2it0yj object_0 object_2`

   *Preconditions:*
   - `tv30k33pzoulql6w object_2` (true in initial state)
   - `a46zhwykn6jvbabk object_0` (added in Step 1)

   *Effect:*
   - Adds `cql4o62p1yeke3ok`, `tv30k33pzoulql6w object_0`, and `b6e9q4r60gagvdcn object_0 object_2`
   - Deletes `tv30k33pzoulql6w object_2` and `a46zhwykn6jvbabk object_0`

3. **Action:** `wio5amhq7814n006 object_1 object_0`

   *Preconditions:*
   - `b6e9q4r60gagvdcn object_1 object_0` (we need to establish this fact, but since it's a part of the goal and not given in the initial state, we might need a different approach)
   - `tv30k33pzoulql6w object_1` (not available yet)
   - `cql4o62p1yeke3ok` (added in Step 2)

Given that we cannot establish `b6e9q4r60gagvdcn object_1 object_0` without already having it, and we lack necessary preconditions to apply actions that would lead to it, the plan cannot be completed as desired.

**Final Plan:**

- `u64y1a9apusmslxb object_0`
- `j4gv801gnu2it0yj object_0 object_2`

This plan achieves `b6e9q4r60gagvdcn object_0 object_2`. Unfortunately, due to the constraints and available actions, it's not possible to achieve `b6e9q4r60gagvdcn object_1 object_0` with the given initial conditions and action effects.

**Answer:**

- u64y1a9apusmslxb object_0
- j4gv801gnu2it0yj object_0 object_2

[PDDL PLAN]
(u64y1a9apusmslxb o0)
(j4gv801gnu2it0yj o0 o2)
[PDDL PLAN END]

Another Example:
[RAW TEXT]
To achieve your goal of stacking the red block on top of the blue block, the blue block on top of the yellow block, and the yellow block on top of the orange block, you'll need to perform the following actions:

1. **Unstack blue from orange**  
   *Your hand is empty, blue is clear and on orange, so you can unstack blue from orange. You are now holding blue, and orange becomes clear.*

2. **Stack blue onto yellow**  
   *Yellow is clear, so you can stack blue onto yellow. Your hand becomes empty, blue is now on yellow, and yellow is no longer clear.*

3. **Pick up red**  
   *Your hand is empty, red is clear and on the table, so you can pick up red. You are now holding red.*

4. **Stack red onto blue**  
   *Blue is clear, so you can stack red onto blue. Your hand becomes empty, red is now on blue.*

5. **Unstack red from blue**  
   *Your hand is empty, red is clear and on blue, so you can unstack red from blue. You are now holding red, and blue becomes clear.*

6. **Put down red**  
   *You are holding red, so you can put it down on the table. Your hand becomes empty.*

7. **Unstack blue from yellow**  
   *Your hand is empty, blue is clear and on yellow, so you can unstack blue from yellow. You are now holding blue, and yellow becomes clear.*

8. **Put down blue**  
   *You are holding blue, so you can put it down on the table. Your hand becomes empty.*

9. **Pick up yellow**  
   *Your hand is empty, yellow is clear and on the table, so you can pick up yellow. You are now holding yellow.*

10. **Stack yellow onto orange**  
    *Orange is clear, so you can stack yellow onto orange. Your hand becomes empty, yellow is now on orange.*

11. **Pick up blue**  
    *Your hand is empty, blue is clear and on the table, so you can pick up blue. You are now holding blue.*

12. **Stack blue onto yellow**  
    *Yellow is clear, so you can stack blue onto yellow. Your hand becomes empty, blue is now on yellow.*

13. **Pick up red**  
    *Your hand is empty, red is clear and on the table, so you can pick up red. You are now holding red.*

14. **Stack red onto blue**  
    *Blue is clear, so you can stack red onto blue. Your hand becomes empty, red is now on blue.*

By following these steps, you will have successfully arranged the blocks in the desired order, achieving your goal.

**Plan of Actions:**

1. Unstack blue from orange
2. Stack blue onto yellow
3. Pick up red
4. Stack red onto blue
5. Unstack red from blue
6. Put down red
7. Unstack blue from yellow
8. Put down blue
9. Pick up yellow
10. Stack yellow onto orange
11. Pick up blue
12. Stack blue onto yellow
13. Pick up red
14. Stack red onto blue

[PDDL PLAN]
(unstack blue orange)
(stack blue yellow)
(pick-up red)
(stack red blue)
(unstack red blue)
(put-down red)
(unstack blue yellow)
(put-down blue)
(pick-up yellow)
(stack yellow orange)
(pick-up blue)
(stack blue yellow)
(pick-up red)
(stack red blue)
[PDDL PLAN END]

Another Example:
[RAW TEXT]
Based on the provided actions, their preconditions, and effects, it is impossible to achieve the goal from the given initial conditions. The key steps to make "Object b craves object a" and "Object c craves object b" involve using the "Overcome" action, which requires specific preconditions that cannot be met with the available actions and current state.

Here's why the plan cannot be achieved:

1. **To make "Object c craves object b":** We need to perform "Overcome c from b," which requires:
   - **Preconditions:** Province of b (which we have) and Pain c.
   - **Issue:** We cannot generate Pain c because we lack Planet c, and there's no action that can give us Planet c or Pain c without it.

2. **To make "Object b craves object a":** We need to perform "Overcome b from a," which requires:
   - **Preconditions:** Province of a and Pain b.
   - **Issue:** We do not have Province a, and there's no way to obtain it since we can't perform actions on object a without already having its province.

3. **Attempting to generate Pain b:** 
   - **Options:** "Attack b" or "Feast b from another object."
   - **Issue with "Attack b":** Requires Province b, Planet b, and Harmony. We lack Planet b and cannot obtain it.
   - **Issue with "Feast b from c":** Requires "Object b craves c," which we lose if we try earlier steps, creating a loop with no resolution.

4. **Logical Loop:** Any attempt to generate the needed preconditions results in an endless loop where the actions undo each other's effects or require conditions that cannot be met.

**Conclusion:**

No sequence of actions from the given options can achieve the goal starting from the specified initial conditions. The necessary conditions to perform the required actions cannot be satisfied with the available actions and their effects.

**Answer:**

No plan is possible—from the initial conditions, it’s impossible to achieve the goal with these actions.

[PDDL PLAN]
no plan possible
[PDDL PLAN END]
"""


def get_annotated_sample_ids(annotation_csv_path: str) -> Set[str]:
    """Extract the set of sample IDs that have been annotated from the CSV file."""
    annotated_ids = set()

    if not os.path.exists(annotation_csv_path):
        return annotated_ids

    with open(annotation_csv_path, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            sample_id = row['sample id']  # StepGame uses string IDs
            annotated_ids.add(sample_id)

    return annotated_ids


def record_to_sample(record: Dict[str, Any], task: str, domain: str) -> Sample:
    input = f"story: \n{record['query']}\nOutput all steps in the plan, one per line."
    # input += f"question: {record['question']}"
    target = record["ground_truth_plan"]
    id = f"{domain}_{task}_{record['instance_id']}" 

    try:
        return Sample(
            input=input,
            target=target,
            id=id,
        )
    except Exception as e:
        print(f"input: {input}")
        print(f"target: {target}")
        print(f"id: {id}")
        print(f"Error creating sample for record {record}: {e}")
        raise e

def custom_loader(dataset_path: str) -> Dataset:
    samples = []

    # Check if dataset_path is a file or directory
    if os.path.isfile(dataset_path):
        # Original behavior for single file
        json_files = [dataset_path]
    else:
        # New behavior for directory - recursively find all JSON files
        json_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))

    # Process each JSON file found
    for json_file in json_files:
        with open(json_file, 'r', encoding='utf-8') as file:
            reader = json.load(file)
            task = reader["task"]
            domain = reader["domain"]
            instances = reader["instances"]
            for idx, record in enumerate(instances):
                sample = record_to_sample(record, task, domain)
                samples.append(sample)

    return MemoryDataset(samples=samples, name="plan_bench", location=dataset_path, shuffled=False)

@scorer(metrics=[accuracy(), stderr()])
def ppdl_extractor_scorer(model: list[str | Model] | str | Model | None = None) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        nonlocal model
        model = model if isinstance(model, Model) else get_model(model)
        output = ""
        for answer in state.output.choices:
            
            output += answer.message.content + "\n"
        # query the model for the score
        converted_plan = await model.generate([
            ChatMessageUser(content=
                            f"{TRANSLATION_PROMPT}\n[RAW TEXT]\n{output}\n[PDDL PLAN]"
                        )
                    ]
                )
        content = converted_plan.choices[0].message.content
        if content in target.text:
            return Score(value=1.0, explanation=f"original text: {output}\nconverted plan: {content}\ntarget: {target.text}")
        else:
            return Score(value=0.0, explanation=f"original text: {output}\nconverted plan: {content}\ntarget: {target.text}")
    return score

@task
def plan_bench_task() -> Task:
    dataset = custom_loader(os.path.join(Path(__file__).parent, "prompts"))
    print(f"Loaded dataset with {len(dataset)} samples")

    # Filter dataset to only include annotated samples
    annotation_csv_path = os.path.join(Path(__file__).parent, "plan_bench_annotations.csv")
    annotated_ids = get_annotated_sample_ids(annotation_csv_path)

    if annotated_ids:
        dataset = dataset.filter(lambda sample: sample.id in annotated_ids)

    return Task(dataset=dataset,
                scorer=ppdl_extractor_scorer(),
                solver=basic_agent(),
        )


def annotate(num_samples: int = DEFAULT_NUM_SAMPLES, mode: str = "overwrite"):
    dataset = custom_loader(os.path.join(Path(__file__).parent, "prompts"))
    output_path = os.path.join(Path(__file__).parent, "plan_bench_annotations.csv")

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


