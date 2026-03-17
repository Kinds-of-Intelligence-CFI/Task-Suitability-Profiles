# CoQA Conversational Question Answering

This is a conversational question answering evaluation taken from https://github.com/google/BIG-bench/tree/main/bigbench/benchmark_tasks/coqa_conversational_question_answering.

The task evaluates a model's ability to answer questions in a conversational context, where each question may depend on previous questions and answers in the conversation.

## Data Format

The task uses JSON data with the following structure:
```json
{
  "data": [
    {
      "id": "sample_id",
      "story": "Background context/story",
      "questions": [
        {
          "turn_id": 1,
          "input_text": "First question"
        }
      ],
      "answers": [
        {
          "turn_id": 1,
          "input_text": "Answer to first question"
        }
      ],
      "additional_answers": {
        "annotator1": [...],
        "annotator2": [...]
      }
    }
  ]
}
```

## Usage

### Running Evaluation

```bash
# Use default dataset
inspect eval ./Evaluations/CoQA/CoQA_task.py --model openai/gpt-4

# Run with specific parameters
inspect eval ./Evaluations/CoQA/CoQA_task.py --model your-model-name --limit 20 --max-connections 3

# Run with different models for comparison
inspect eval ./Evaluations/CoQA/CoQA_task.py --model anthropic/claude-3-sonnet --limit 10
```

### Python API (Alternative)

If you need programmatic access:

```python
from inspect_ai import eval
from Evaluations.CoQA.CoQA_task import CoQA_task

# Use default dataset
task = CoQA_task()
results = eval(task, model="openai/gpt-4")

# Use custom dataset
task = CoQA_task("path/to/your/coqa_data.json")
results = eval(task, model="your-model-name")
```

## Annotation

To run the annotation workflow for capability assessment:

```python
# Run annotation directly
python ./Evaluations/CoQA/CoQA_task.py
```

This will:
1. Load the CoQA dataset
2. Convert chat messages to string format for annotation
3. Apply OECD capability rubrics to each sample  
4. Save results to `coqa_annotations.csv`

## Scoring

The task uses the `includes()` scorer, which checks if the model's response includes any of the acceptable answers from the target list. This accounts for multiple valid answers provided by different annotators.