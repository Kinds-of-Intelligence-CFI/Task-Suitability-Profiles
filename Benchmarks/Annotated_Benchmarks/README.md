# Evaluations

This folder contains all the code needed to run evaluations for each task, including resources like original JSON files or images.

## Available Tasks

- **CoQA**: Conversational Question Answering task
- **Known_Unknowns**: Task for handling uncertain scenarios

## Running Evaluations

Each task can be run independently using the Inspect AI command line:

```bash
# Run a specific task
inspect eval ./Evaluations/TaskName/task_name.py --model your-model-name

# Run with additional parameters
inspect eval ./Evaluations/TaskName/task_name.py --model openai/gpt-4 --limit 50 --max-connections 5
```

Results are automatically logged to the `logs/` directory with timestamps.

## Adding New Tasks

Follow these steps to add a new evaluation task:

### 1. Create Task Directory

Create a new directory under `Evaluations/` with your task name:

```
Evaluations/
├── YourTaskName/
│   ├── __init__.py
│   ├── your_task_name.py
│   ├── README.md
│   └── data files (JSON, images, etc.)
```

### 2. Implement Task Function

Create your main task file (`your_task_name.py`) following this template:

```python
import json
import os
from pathlib import Path
from typing import Any, Dict

from inspect_ai import Task, task
from inspect_ai.dataset import Dataset, MemoryDataset, Sample
from inspect_ai.scorer import includes  # or appropriate scorer

def custom_loader(dataset_path: str) -> Dataset:
    """Load your dataset and convert to Inspect AI format"""
    # Load your data (JSON, CSV, etc.)
    data = json.load(open(dataset_path, 'r'))
    
    samples = []
    for item in data:
        sample = Sample(
            input=item["question"],  # Adapt to your data structure
            target=item["answer"],   # Expected answer(s)
            id=item["id"],          # Unique identifier
            metadata={              # Additional metadata
                "source": item.get("source", ""),
                # Add other relevant fields
            }
        )
        samples.append(sample)
    
    return MemoryDataset(
        samples=samples, 
        name="YourTaskName", 
        location=dataset_path, 
        shuffled=False
    )

@task
def your_task_name(dataset_path: str | None = None) -> Task:
    """Main task function"""
    if dataset_path is None:
        dataset_path = os.path.join(Path(__file__).parent, "default_data.json")
    
    dataset = custom_loader(dataset_path)
    
    return Task(
        dataset=dataset,
        scorer=includes(),  # Choose appropriate scorer
        # Add solver if needed for multi-step tasks
    )

# Optional: Add annotation support
def convert_input_to_string(dataset: Dataset) -> Dataset:
    """Convert inputs to strings for annotation (if needed)"""
    for sample in dataset:
        # Convert complex inputs (like chat messages) to strings
        sample.input = str(sample.input)
    return dataset

# Optional: Main block for direct execution
if __name__ == "__main__":
    from Annotations.annotate_tasks import annotate_task, extract_annotations
    from inspect_ai import eval
    
    # Run annotation
    dataset_path = os.path.join(Path(__file__).parent, "data.json")
    output_path = os.path.join(Path(__file__).parent, "annotations.csv")
    
    dataset = custom_loader(dataset_path)
    dataset = convert_input_to_string(dataset)
    
    annotation_task = annotate_task(dataset)
    log = eval(annotation_task, model="openai/azure/gpt-4o" )
    extract_annotations(log[0], output_path)
```

