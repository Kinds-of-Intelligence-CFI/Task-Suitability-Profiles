# Benchmarks

This folder contains all code required to annotate benchmarks with cognitive capability demands and evaluate AI models on them. It is the first stage of the Task Suitability pipeline.

## Available Benchmarks

The suite covers 25 benchmarks spanning the 18 cognitive capabilities used in the profiling framework. Pre-computed annotations (by `openai/gpt-4o`) are included in each benchmark directory as `annotations.csv`.

| Benchmark | Primary Capabilities | Description |
|-----------|----------------------|-------------|
| **AGIEval** | Language, Causal Reasoning, Working Memory | College entrance exam questions (SAT, LSAT, GRE-style) requiring expert reasoning |
| **Abstract Narrative Understanding** | Language, Episodic Memory, Cognitive Flexibility | Classifying the abstract structure of short narratives |
| **BigBenchHard** | Multi-capability | 23 challenging reasoning sub-tasks from the BIG-Bench suite |
| **BigToM** | Theory of Mind, Working Memory | Multi-agent scenarios requiring belief tracking across complex conditions |
| **Cause and Effect** | Causal Reasoning | Identifying causes and effects from short scenario descriptions |
| **CoQA** | Language, Episodic Memory | Conversational question answering over diverse reading passages |
| **Crow** | Theory of Mind, Language | Commonsense reasoning over dialogue (intent, stance, safety, summarisation) |
| **EmoBench** | Emotion Perception and Empathy, Theory of Mind | Emotional intelligence and situational empathy tasks |
| **EWoK** | Mental Simulation, Causal Reasoning | Embodied world knowledge tasks probing physical and social intuitions |
| **Evaluating Information Essentiality** | Attention and Inhibitory Control, Working Memory | Identifying which information is essential to a given decision |
| **Fantasy Reasoning** | Cognitive Flexibility, Causal Reasoning | Reasoning correctly under counterfactual or fantasy-world constraints |
| **FanToM** | Theory of Mind, Episodic Memory | Theory of mind in conversational contexts with partial information |
| **INTUIT** | Mental Simulation, Causal Reasoning | Intuitive physics and causal prediction from the VIGNET benchmark suite |
| **Known Unknowns** | Metacognition | Recognising the limits of one's own knowledge |
| **LLM BabyBench** | Planning, Working Memory | Basic cognitive decomposition tasks adapted from developmental psychology |
| **MacGyver** | Planning, Mental Simulation, Cognitive Flexibility | Creative problem-solving using only a limited set of available objects |
| **MetaMedQA** | Metacognition, Semantic Memory | Medical question answering with explicit uncertainty and confidence awareness |
| **NarrativeQA** | Episodic Memory, Language | Reading comprehension over long narrative documents (books and film scripts) |
| **NEWTON** | Causal Reasoning, Mental Simulation | Newtonian physics and physical intuition tasks |
| **OpenTOM** | Theory of Mind, Language | Open-ended theory of mind questions about characters' beliefs and intentions |
| **Plan Bench** | Planning, Working Memory | Multi-step planning and action sequencing tasks |
| **SocialNorm** | Theory of Mind, Emotion Perception and Empathy | Social norm understanding and moral judgement |
| **StepGame** | Spatial Reasoning and Navigation | Step-by-step spatial direction following to determine relative positions |
| **Text Navigation** | Spatial Reasoning and Navigation, Planning | Navigation through text-based maze environments using a stateful tool |
| **Tiger MMLU** | Semantic Memory, Language | Expert-level academic knowledge across 57 subject domains (MMLU-Pro) |

## Dataset Access

Most benchmarks use data bundled in this repository. The following require additional setup:

**Download from GitHub:**
- **INTUIT** — download `battery_for_ai_clean.csv` from [VIGNET](https://github.com/Kinds-of-Intelligence-CFI/VIGNET) and place it at `Benchmarks/Annotated_Benchmarks/INTUIT/battery_for_ai_clean.csv`

**Requires a Hugging Face account** — log in (`huggingface-cli login`) and accept the terms of use for each dataset:
- [ZhengyanShi/StepGame](https://huggingface.co/datasets/ZhengyanShi/StepGame) — StepGame
- [NEWTONReasoning/NEWTON](https://huggingface.co/datasets/NEWTONReasoning/NEWTON) — NEWTON
- [ewok-core/ewok-core-1.0](https://huggingface.co/datasets/ewok-core/ewok-core-1.0) — EWoK
- [salem-mbzuai/LLM-BabyBench](https://huggingface.co/datasets/salem-mbzuai/LLM-BabyBench) — LLM BabyBench
- [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro) — Tiger MMLU
- [deepmind/narrativeqa](https://huggingface.co/datasets/deepmind/narrativeqa) — NarrativeQA
- [maximegmd/MetaMedQA](https://huggingface.co/datasets/maximegmd/MetaMedQA) — MetaMedQA
- [socialnormdataset/social](https://huggingface.co/datasets/socialnormdataset/social) — SocialNorm

Benchmarks without these dependencies can be run immediately after installation.

## Annotating Benchmarks

Annotation rates each benchmark item against the 18-capability rubric using an LLM. Pre-computed annotations by `openai/gpt-4o` are already included — you only need to re-run this step if you want to use a different annotator or add new benchmarks.

```bash
python -m Benchmarks.Annotations.run_annotations --model <model_name>
# e.g.
python -m Benchmarks.Annotations.run_annotations --model openai/gpt-4o
```

Annotations are written to `annotations.csv` within each benchmark's directory. The rubric is defined in `Benchmarks/Annotations/rubric.json` (compiled from the per-capability rubric files in `rubric_files/`). The number of items annotated per benchmark is controlled by `Benchmarks/Annotations/item_allocations.json`.

## Evaluating a Model

To evaluate a model across all benchmarks and produce a single results CSV:

```bash
python -m Benchmarks.run_all_tasks --model <model_name>
# e.g.
python -m Benchmarks.run_all_tasks --model openai/gpt-4o-mini
```

Results are saved to `./logs/` as a CSV. Pre-computed results for `openai/gpt-4o-mini` are available at `Suitability/data/raw/gpt-4o-mini_results.csv` if you want to skip this step.

You can also run a specific benchmark directly using Inspect AI:

```bash
inspect eval Benchmarks/Annotated_Benchmarks/<BenchmarkName>/<task_file>.py --model <model_name>
```

Additional Inspect AI arguments (e.g. `--limit`, `--max-connections`) are passed through by `run_all_tasks`. To evaluate only a subset of benchmarks:

```bash
python -m Benchmarks.run_all_tasks --model openai/gpt-4o --include coqa_task bigtom_task
```

To preview which tasks would run without executing them:

```bash
python -m Benchmarks.run_all_tasks --model openai/gpt-4o --dry-run
```

## Benchmarking Local Model Speed

Local Hugging Face models run through Inspect's `hf/` provider (transformers) or
`vllm/` provider, so no task changes are needed -- only the model string changes and
the backend must be installed. The runner can also measure throughput and peak memory.

### Install

The inference backends (`transformers` for `hf/`, `vllm` for `vllm/`) ship as part of
the main dependencies, so a normal `uv sync` installs them. Note that `vllm` requires a
CUDA GPU, so on a CPU/Apple-MPS laptop use only the `hf/` provider.

```bash
uv sync
```

### Run with timing

Pass `--timing` to sample peak RAM/VRAM during each eval and extract throughput. Pin
`--limit` and `--max-tokens` so each machine does the same amount of work:

```bash
# Laptop (transformers, sequential for clean per-sample numbers)
uv run -m Benchmarks.run_all_tasks --model hf/Qwen/Qwen2.5-1.5B-Instruct \
  --include coqa_task tiger_mmlu_task --limit 50 \
  --max-connections 1 --max-tokens 256 --timing -M device=mps

# Server (vLLM, concurrent for realistic throughput)
uv run -m Benchmarks.run_all_tasks --model vllm/Qwen/Qwen2.5-1.5B-Instruct \
  --include coqa_task tiger_mmlu_task --limit 50 \
  --max-tokens 256 --timing -M gpu_memory_utilization=0.9
```

`-M` passes model arguments through to the provider (e.g. `device`, `gpu_memory_utilization`).

### Outputs

In addition to the usual `results.csv`, `--timing` writes to the log directory:

- `timing_summary.csv` -- one row per task: `eval_wall_seconds`, `total_output_tokens`,
  `aggregate_tokens_per_sec`, plus mean/p50/p95 of per-sample tokens/sec and working time.
- `timing_per_sample.csv` -- per-sample `total_time`, `working_time`, token counts, and
  `output_tokens_per_sec`.
- `memory.csv` -- per-task `peak_ram_mb` and `peak_vram_mb` (VRAM is blank without an
  NVIDIA GPU; RAM requires `psutil`, included in the `local` extra).

### Fairness caveats

- Use **`aggregate_tokens_per_sec`** (total output tokens over the eval wall-clock) for
  cross-machine comparison. Under vLLM batching the per-sample `working_time` windows
  overlap, so the per-sample tokens/sec stats are only meaningful with
  `--max-connections 1` (sequential).
- `--max-tokens` caps generation length so throughput is not dominated by some samples
  happening to generate much longer outputs than others.
- VRAM is sampled per-process via `nvidia-smi`; close other GPU jobs for clean numbers.

## Adding a New Benchmark

### 1. Create the benchmark directory

```
Benchmarks/Annotated_Benchmarks/
└── YourBenchmark/
    ├── __init__.py
    ├── your_benchmark_task.py
    └── data/                  # local data files if needed
```

### 2. Implement the task file

Each benchmark must define an Inspect AI `@task` function and a `convert_input_to_string` helper used by the annotation pipeline:

```python
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.scorer import exact  # or appropriate scorer
from inspect_ai.solver import generate

@task
def your_benchmark_task() -> Task:
    samples = [
        Sample(input="Your question here", target="Expected answer", id="item_001"),
        # ...
    ]
    return Task(
        dataset=MemoryDataset(samples, name="YourBenchmark"),
        solver=[generate()],
        scorer=exact(),
    )

def convert_input_to_string(dataset):
    """Required for the annotation pipeline — ensures inputs are plain strings."""
    for sample in dataset:
        if not isinstance(sample.input, str):
            sample.input = str(sample.input)
    return dataset
```

### 3. Register the task in `run_all_tasks.py`

Add an entry to the `TASKS` dict in `Benchmarks/run_all_tasks.py`:

```python
"your_benchmark_task": {
    "file": "Benchmarks/Annotated_Benchmarks/YourBenchmark/your_benchmark_task.py",
    "function": "your_benchmark_task",
},
```

### 4. Set an annotation budget

Add an entry to `Benchmarks/Annotations/item_allocations.json`:

```json
"your_benchmark_annotation": 200
```

This controls how many items are sampled for annotation (0 = skip annotation for this benchmark).
