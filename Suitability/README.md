# Task-Ability Profiles

A capability-based profiling system for evaluating agent performance on cognitive tasks using Bayesian Item Response Theory.

## Overview

This project provides a framework for:

1. **Defining cognitive abilities** - A standardized set of 18 cognitive abilities (e.g., Working Memory, Planning, Theory of Mind)
2. **Mapping task demands** - Quantifying how important each ability is for different work tasks
3. **Collecting human expertise** - Processing questionnaire data to build empirical ability demand matrices
4. **Simulating agent performance** - Creating synthetic agents with known capability profiles
5. **Inferring capability profiles** - Using Bayesian inference to estimate agent capabilities from performance data
6. **Computing suitability scores** - Matching agent capabilities to task demands

## Installation

```bash
# Clone the repository
git clone https://github.com/jonnyp1990/task-ability-profiles.git
cd task-ability-profiles

# Install dependencies
pip install -r requirements.txt

# Or install as a package
pip install -e .
```

## Project Structure

```
task-ability-profiles/
├── src/                          # Source code
│   ├── core/                     # Core modules
│   │   ├── capabilities.py       # Capability name standardization & validation
│   │   ├── model.py              # Bayesian model building & fitting
│   │   └── visualization.py      # Radar plots, ICC curves, etc.
│   ├── pipeline/                 # Pipeline modules
│   │   ├── simulation.py         # Simulate agent performance
│   │   ├── inference.py          # Run Bayesian inference
│   │   ├── suitability.py        # Compute suitability scores
│   │   └── questionnaire.py      # Process questionnaire data
│   └── utils/                    # Utilities
│       └── io.py                 # Data loading & saving
├── scripts/                      # Runner scripts
│   ├── run_sim_inference.py      # Simulate + infer with identical model parameters (recommended for validation)
│   ├── run_simulation.py         # Simulate agent data only
│   ├── run_inference.py          # Fit capability models to existing performance data
│   ├── run_population_inference.py  # Fit hierarchical model to multiple human participants
│   ├── run_suitability.py        # Compute task suitability
│   ├── visualize_profiles.py     # Generate visualizations
│   ├── build_ability_matrix.py   # Build ability matrices from questionnaires
│   ├── analyze_battery_coverage.py  # Analyse annotation battery coverage and identifiability
│   └── select_battery_items.py   # Select stratified item subset for human testing
├── config/                       # Configuration files
│   ├── abilities.csv             # 18 cognitive abilities definition
│   └── tasks.csv                 # Work task definitions
├── questionnaire/                # Questionnaire data & analysis
│   ├── *.csv                     # Raw questionnaire response data
│   └── analyse_questionnaire_data.ipynb  # Analysis notebook
├── data/                         # Data directory
│   ├── raw/                      # Raw input data
│   ├── processed/                # Processed data (including ability matrices)
│   └── results/                  # Model outputs
├── figures/                      # Generated figures
├── notebooks/                    # Jupyter notebooks
└── tests/                        # Unit tests
```

## The Capability Framework

### 18 Cognitive Abilities

| # | Ability | Acronym | Description |
|---|---------|---------|-------------|
| 1 | Episodic Memory | EM | Remembering previous events |
| 2 | Semantic Memory | SM | Remembering facts and information |
| 3 | Procedural Memory | ProcM | Remembering how to perform learned tasks |
| 4 | Prospective Memory | ProsM | Remembering to do planned actions |
| 5 | Working Memory | WM | Holding multiple pieces of information at once |
| 6 | Attention and Inhibitory Control | AaIC | Controlling behaviors to focus on tasks |
| 7 | Cognitive Flexibility | CF | Switching between tasks or adapting |
| 8 | Mental Simulation | MS | Imagining possible future scenarios |
| 9 | Planning | P | Mapping out strategies to achieve goals |
| 10 | Metacognition | MC | Assessing own thoughts and performance |
| 11 | Perception and Pattern Recognition | PaPR | Noticing relevant details |
| 12 | Functional Perception | FP | Recognizing appropriate roles for objects/people |
| 13 | Spatial Reasoning and Navigation | SRaN | Reasoning about size, space, and distance |
| 14 | Object Permanence | OP | Understanding objects exist when not visible |
| 15 | Causal Reasoning | CR | Understanding cause and effect |
| 16 | Theory of Mind | ToM | Reasoning about others' goals and beliefs |
| 17 | Emotion Perception and Empathy | EPaE | Connecting with others' feelings |
| 18 | Language | L | Using language to communicate |

### Work Domains

The questionnaire covers 6 work domains:

| # | Type | Sub-type | Acronym |
|---|------|----------|---------|
| 1 | Manual-physical | Warehouse or logistics | WL |
| 2 | Manual-physical | Manufacture, maintenance, or repair | MMR |
| 3 | Computer-digital | Numerical, data, or programming | NDP |
| 4 | Computer-office | Admin, organisational, or planning | AOP |
| 5 | Computer-office | Customer service, marketing or HR | CMH |
| 6 | Face-to-face | Hospitality, sales, or client care | HSC |

## Usage

### 1. Validate Model with Simulation (recommended starting point)

Use `run_sim_inference.py` to simulate agent performance and immediately fit the same model, guaranteeing that the generative model and the inference model share identical parameters. The script prints a recovery table comparing true vs. estimated capability levels.

```bash
# Normalized additive pooling (recommended)
python scripts/run_sim_inference.py \
    --pool add --normalize \
    --output data/results/sim_normadd

# Single agent, save the simulated performance data too
python scripts/run_sim_inference.py \
    --pool add --normalize \
    --agents social_specialist \
    --save-performance \
    --output data/results/sim_test

# Softmin pooling — use higher mu-c to compensate for lower logit scale
python scripts/run_sim_inference.py \
    --pool softmin --tau 1.0 --mu-c 4.0 \
    --output data/results/sim_softmin
```

Example recovery output:
```
  social_specialist  (baseline c=1.0)
    Capability                          True    Est   Error
    ---------------------------------------------------------
    Theory of Mind                       5.0   5.43   +0.43
    Language                             5.0   4.73   -0.27
    [baseline capabilities]              1.0   2.08   mae=1.08
```

### 2. Build Ability Matrices from Questionnaire Data

Create ability demand matrices from questionnaire responses:

```bash
# Build combined ability matrix (all participants, all domains)
python scripts/build_ability_matrix.py \
    --companies questionnaire/Future_of_skills_companies_20260112.csv \
    --online questionnaire/Future_of_skills_online_20260112.csv \
    --output data/processed/ability_matrix_combined.csv

# Build for a specific domain (e.g., domain 3 = NDP)
python scripts/build_ability_matrix.py \
    --companies questionnaire/Future_of_skills_companies_20260112.csv \
    --online questionnaire/Future_of_skills_online_20260112.csv \
    --domain 3 \
    --output data/processed/ability_matrix_NDP.csv \
    --plot

# Build from one data source only
python scripts/build_ability_matrix.py \
    --online questionnaire/Future_of_skills_online_20260112.csv \
    --source online \
    --output data/processed/ability_matrix_online.csv

# Compare ability matrices between data sources
python scripts/build_ability_matrix.py \
    --companies questionnaire/Future_of_skills_companies_20260112.csv \
    --online questionnaire/Future_of_skills_online_20260112.csv \
    --compare-sources \
    --output data/processed/ability_matrix

# Build matrices for all domains separately
python scripts/build_ability_matrix.py \
    --companies questionnaire/Future_of_skills_companies_20260112.csv \
    --online questionnaire/Future_of_skills_online_20260112.csv \
    --all-domains \
    --output data/processed/ability_matrix
```

### 3. Simulate Agent Performance (standalone)

Generate synthetic performance data only (use `run_sim_inference.py` instead when you also want to fit the model):

```bash
# Normalized additive pooling
python scripts/run_simulation.py \
    --pool add --normalize \
    --output data/processed/simulated_normadd

# Softmin pooling
python scripts/run_simulation.py \
    --pool softmin --tau 1.0 \
    --output data/processed/simulated_softmin
```

### 4. Run Bayesian Inference on Real Data

Fit capability models to existing performance data. The pooling method must match the assumed generative model.

```bash
# LLM evaluation data — normalized additive pooling (recommended)
python scripts/run_inference.py \
    --mode llm \
    --annotations data/processed/annotations.csv \
    --results data/raw/gpt-4o-mini_results.csv \
    --agent-name gpt-4o-mini \
    --pool add --normalize \
    --output data/results/llm_normadd

# Infer only a subset of capabilities (acronyms from abilities.csv)
# --normalize keeps estimates stable regardless of subset size
python scripts/run_inference.py \
    --mode llm \
    --annotations data/processed/annotations.csv \
    --results data/raw/gpt-4o-mini_results.csv \
    --agent-name gpt-4o-mini \
    --pool add --normalize \
    --capabilities EM SM WM AaIC L MC \
    --output data/results/llm_subset
```

### 5. Compute Suitability Scores

Match agent capabilities to task demands. `--ability-matrix` is required and accepts one matrix per run; run separately for each matrix you want to score against.

```bash
# Combined ability matrix (all domains, all participants)
python scripts/run_suitability.py \
    --agents strong_generalist weak_generalist social_specialist \
    --idata-base data/results/simulated \
    --abilities config/abilities.csv \
    --tasks config/tasks.csv \
    --ability-matrix data/processed/ability_matrix_combined.csv \
    --output figures/suitability_combined.png \
    --column-normalize --no-show

# Domain-specific matrix (e.g. NDP)
python scripts/run_suitability.py \
    --agents strong_generalist weak_generalist social_specialist \
    --idata-base data/results/simulated \
    --abilities config/abilities.csv \
    --tasks config/tasks.csv \
    --ability-matrix data/processed/ability_matrix_combined_domain_NDP.csv \
    --output figures/suitability_NDP.png \
    --column-normalize --no-show

# Recommended power and scale settings
# --power 0.5: square-root mean — partially compensatory, robust to noisy weights
# --use-ratio: uses theta = exp(c) (ratio-scale, strictly positive) for numeric stability
python scripts/run_suitability.py \
    --agents gpt-4o-mini \
    --idata-base data/results/llm_normadd \
    --abilities config/abilities.csv \
    --tasks config/tasks.csv \
    --ability-matrix data/processed/ability_matrix_combined.csv \
    --output figures/suitability_llm.png \
    --column-normalize --power 0.5 --use-ratio --no-show
```

**`--column-normalize`** (recommended): min-max normalises each capability column across tasks before computing demand weights. The raw ability matrix tends to have near-uniform demand levels across tasks (most abilities rated moderately-to-highly for most tasks), which collapses demand weights toward 1/K and produces flat scores regardless of the agent's profile. Column normalisation converts absolute demand to *relative* demand — how much does this task emphasise this ability compared to other tasks — making scores sensitive to the match between the agent's specific capability pattern and each task's specific demands.

**`--power`**: Controls compensability of the weighted power mean. `p=0` is the geometric mean (moderate compensability); `p=0.5` (square-root mean) is empirically robust for importance-weighted ability matrices; `p=1` is arithmetic mean; negative `p` is non-compensatory but noisy — use `--use-ratio` if `p < 0`.

**`--use-ratio`**: Uses ratio-scale capabilities `theta = exp(c)` instead of log-scale `c`. Required for numeric stability when `p < 0`. Recommended for `p = 0.5` with softmin-inferred profiles.

**Note on cross-matrix comparability**: column normalisation is fitted to the tasks present in the matrix passed in. Scores from different domain-specific matrices are therefore not directly comparable on the same scale.

### 6. Analyse Battery Coverage (for human testing design)

Before selecting items for a human battery, analyse the annotated item pool to understand coverage and identifiability:

```bash
# Default analysis (excludes ProsM, OP, EPaE; extends rank curve to 500 items)
python scripts/analyze_battery_coverage.py \
    --annotations data/processed/annotations.csv \
    --abilities config/abilities.csv \
    --output figures/battery_coverage \
    --no-show

# Custom exclusions and target size
python scripts/analyze_battery_coverage.py \
    --annotations data/processed/annotations.csv \
    --abilities config/abilities.csv \
    --exclude ProsM OP EPaE \
    --target-n 150 \
    --output figures/battery_coverage \
    --save-table \
    --no-show
```

Produces five plots: capability coverage bar chart, co-occurrence heatmap, demand distribution violin, item dimensionality histogram, and a rank/min-singular-value curve (random vs greedy vs target-n).

### 7. Select Human Battery Items

Select a stratified subset of items optimised for identifiability across capabilities and demand levels:

```bash
# 150-item battery (default: excludes ProsM, OP, EPaE)
python scripts/select_battery_items.py \
    --annotations data/processed/annotations.csv \
    --abilities config/abilities.csv \
    --exclude ProsM OP EPaE \
    --target-n 150 \
    --output data/processed/selected_battery \
    --no-show
```

**Selection strategy:**
1. **Stratified phase**: for each capability × demand-level cell (D=1..5), sample up to `--items-per-level` items where that capability is the primary demand.
2. **E-optimal fill**: greedily add remaining items by maximising the minimum singular value of the demand sub-matrix (targets the weakest-covered capability direction).

Output: `selected_battery_items.csv` with columns `[dataset name, sample id, primary_capability, primary_d_level, ...]` plus plots of difficulty spread and identifiability vs. random.

**Performance CSV format for human data** (participants × items):
```
participant_id, dataset1_item001, dataset1_item002, dataset2_item003, ...
P001,           1,               0,               1, ...
P002,           NaN,             1,               0, ...
```
Column names must be formatted as `{dataset_name}_{sample_id}` to align with the annotations file.

### 8. Run Population (Hierarchical) Inference on Human Data

Fit a hierarchical IRT model to multiple participants simultaneously. Partial pooling across participants gives tighter individual estimates than fitting each person independently.

```bash
# Additive pooling (recommended)
python scripts/run_population_inference.py \
    --performance data/processed/human_performance.csv \
    --annotations data/processed/selected_battery_items.csv \
    --output data/results/population \
    --pool add --normalize

# Softmin pooling (non-compensatory)
python scripts/run_population_inference.py \
    --performance data/processed/human_performance.csv \
    --annotations data/processed/selected_battery_items.csv \
    --output data/results/population_softmin \
    --pool softmin --tau 0.25 --mu-c 2.5

# With radar plot of individual profiles
python scripts/run_population_inference.py \
    --performance data/processed/human_performance.csv \
    --annotations data/processed/selected_battery_items.csv \
    --output data/results/population \
    --pool add --normalize \
    --radar-output figures/population_radar.png \
    --no-show
```

Outputs:
- `{output}_population.nc`: ArviZ InferenceData (population parameters + per-participant offsets)
- `{output}_population_meta.json`: participant names and capability column order

### 9. Visualize Results

Generate capability radar plots and diagnostics:

```bash
python scripts/visualize_profiles.py \
    --agents gpt-4o-mini \
    --idata-base data/results/llm_normadd \
    --abilities config/abilities.csv \
    --output figures/ \
    --plot-forest
```

## The Bayesian Model

The model uses Measurement Layouts (https://arxiv.org/abs/2309.11975) to infer latent capability levels from binary performance data.

### Model Structure

For agent *i* on item *j* with capability dimension *k*:

```
c[k] ~ Normal(mu_c, sigma_c)           # Log-scale capability: c[k] = log(theta[k])
theta[k] = exp(c[k])                   # Ratio-scale capability (primary quantity of interest)
kappa[k] ~ HalfNormal(sigma_kappa)     # Discrimination weight (fixed to 1 by default)
alpha ~ Normal(mu_alpha, sigma_alpha)  # Intercept

delta[j,k] = exp(lambda * D[j,k])     # Ratio-scale difficulty for item j, ability k
margin[j,k] = c[k] - lambda * D[j,k]  # log(theta[k] / delta[j,k]): log-ratio of capability
                                       # to difficulty; 0 if D[j,k] = 0 (ability not required)
z[j] = f(margins)                      # Log-odds (depends on pooling method)
p[j] = sigmoid(z[j])                   # Success probability
Y[j] ~ Bernoulli(p[j])                 # Observed outcome
```

The conceptually primary quantities are `theta[k]` (capability) and `delta[j,k]` (difficulty) on a ratio scale — both have an absolute zero and meaningful multiplicative structure. Inference is parameterised in log-space via `c[k]` for tractability. The margin is therefore a log-ratio: positive when capability exceeds difficulty (`theta[k] > delta[j,k]`), negative when it falls short. The ratio-scale framing is conceptual; it does not constrain margins to be positive.

### Pooling Methods

The pooling method determines how margins across capabilities are combined into a single log-odds. The choice reflects a theoretical assumption about how capabilities interact:

| Method | Formula | Behaviour | When to use                                                                                                                                                                                                                                       |
|--------|---------|-----------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `add` + `normalize` | `z = alpha + mean_k(margin[j,k])` | Compensatory — capabilities substitute for each other. Scale-invariant to number of active capabilities. | **Recommended.** Correctly recovers full range of capability levels from simulation; estimates are stable across capability subsets.                                                                                                              |
| `softmin` | `z = -(logsumexp(-τ·margin) - log(K)) / τ` | Non-compensatory — weakest capability limits success. As τ→0, approaches `add + normalize`; as τ→∞, approaches hard minimum. | Appropriate when a single capability bottleneck is theoretically motivated. Note: capabilities above the bottleneck contribute zero gradient, causing wide posteriors and underestimation of specialist profiles. Can use higher `--mu-c` (4.0+). |
| `add` | `z = alpha + sum_k(kappa[k] * margin[j,k])` | Compensatory but logit scale grows with number of active capabilities. | Avoid for multi-capability tasks; use `normalize` instead.                                                                                                                                                                                        |
| `geom` | `z = (logsumexp(τ·margin) - log(K)) / τ` | Strongest capability dominates. | Experimental.                                                                                                                                                                                                                                     |

**On the softmin/normadd relationship:** The softmin formula is a generalized mean on the exponentiated margin scale — `tau=0` (limit) is exactly `normalize`, `tau=1` is the harmonic mean of exp(margins). Reducing `tau` moves the model toward compensatory pooling and improves recovery of high capability levels.

### Key Parameters

- **theta[k]**: Ratio-scale capability for ability k — the primary quantity of interest
- **c[k]**: Log-scale capability, `c[k] = log(theta[k])`; used for inference (Normal prior)
- **D[j,k]**: Demand level (0-5) for ability k on item j
- **delta[j,k]**: Ratio-scale difficulty, `delta[j,k] = exp(lambda * D[j,k])`
- **lambda**: Per-level log step; each +1 demand step multiplies raw difficulty by `exp(lambda)`
- **kappa[k]**: Discrimination weight (how much ability k contributes to overall performance)
- **alpha**: Intercept (baseline log-odds when capabilities match demands)
- **normalize**: If true, divides additive sum by number of active capabilities per item

## Suitability Scoring

Suitability scores combine inferred capabilities with task demand profiles using a weighted power mean:

```
S_task = (sum_k w[k] * c[k]^p)^(1/p)
```

Where:
- **w[k]**: Demand weights for task (from ability matrix, optionally column-normalised)
- **c[k]**: Posterior capability samples
- **p**: Power parameter (p=1: arithmetic, p=2: quadratic mean)

Uncertainty is incorporated via:
- Posterior sampling from inferred capabilities
- Dirichlet sampling over demand weights

### Column normalisation (`--column-normalize`)

The raw ability matrix often has near-uniform demand levels (most abilities rated moderately-to-highly for most tasks). After normalising weights to sum to 1, each weight is close to 1/K, so the weighted mean of c[k] is nearly the same across all tasks regardless of the agent's profile.

Column normalisation addresses this by min-max scaling each capability column across tasks to [0, 1] before computing weights:

```
D_norm[t, k] = (D[t, k] - min_t(D[t, k])) / (max_t(D[t, k]) - min_t(D[t, k]))
```

This converts absolute demand into *relative* demand — how much does this task emphasise this ability compared to other tasks. An ability with low cross-task variance (e.g. EM, std=1.4) contributes little discriminating power; an ability with high variance (e.g. EPaE, std=6.2, range 0–28) differentiates tasks strongly. The result is suitability scores that reflect the match between the agent's specific capability pattern and each task's specific demands.

## Python API

### Building Ability Matrices

```python
from src.pipeline.questionnaire import (
    load_questionnaire_data,
    apply_quality_control,
    build_ability_matrix,
    build_ability_matrices_by_domain,
    compare_ability_matrices,
)

# Load questionnaire data
df, tasks_df, abilities_df, domains_df = load_questionnaire_data(
    companies_path="questionnaire/Future_of_skills_companies_20260112.csv",
    online_path="questionnaire/Future_of_skills_online_20260112.csv",
    tasks_path="config/tasks.csv",
    abilities_path="config/abilities.csv",
)

# Apply quality control
df = apply_quality_control(df, completion_threshold=100, duration_threshold=600)

# Build ability matrix for all data
ability_matrix = build_ability_matrix(df, tasks_df, abilities_df)

# Build for specific domain
ability_matrix_ndp = build_ability_matrix(
    df, tasks_df, abilities_df,
    source="combined",
    domain=3,  # NDP domain
)

# Build matrices for all domains
matrices_by_domain = build_ability_matrices_by_domain(df, tasks_df, abilities_df)
```

### Simulation and Inference

```python
from src.utils.io import load_abilities, load_annotations
from src.pipeline.simulation import create_simulated_data
from src.pipeline.inference import run_inference_batch
from src.core.model import collect_capability_means
from src.core.visualization import plot_radar_capabilities

# Load data
abilities_df = load_abilities("config/abilities.csv")
annotations_df, capability_cols, D = load_annotations("data/processed/annotations.csv")

# Simulate agents — pooling parameters must match inference below
C_df, Ym_df, Pm_df = create_simulated_data(
    capability_cols, D, pool="add", normalize=True, seed=42
)

# Run inference with identical pooling parameters
results = run_inference_batch(
    agents=list(C_df.index),
    performance_df=Ym_df,
    demand_matrix=D,
    item_index=Ym_df.columns.values,
    output_base_path="data/results/sim_normadd",
    pool="add",
    normalize=True,
    mu_c=3.0,
)

# Visualize
capability_df = collect_capability_means(
    {name: idata for name, (idata, _) in results.items()},
    capability_cols,
)
plot_radar_capabilities(capability_df, abilities_df, save_path="figures/radar.png")
```

### Population (Hierarchical) Inference

```python
from src.utils.io import load_abilities, load_population_idata
from src.pipeline.inference import run_population_inference
from src.core.model import extract_population_capability_samples, collect_capability_means
from src.core.visualization import plot_radar_capabilities
import pandas as pd

# Load participant performance (rows=participants, columns={dataset}_{item_id}, values=0/1/NaN)
perf_df = pd.read_csv("data/processed/human_performance.csv", index_col=0)

abilities_df = load_abilities("config/abilities.csv")
annotations_df, capability_cols, D = load_annotations("data/processed/selected_battery_items.csv")

# Fit hierarchical model (partial pooling across participants)
idata, model = run_population_inference(
    performance_df=perf_df,
    demand_matrix=D,
    item_index=perf_df.columns.values,
    capability_cols=capability_cols,
    output_base_path="data/results/population",
    pool="add",
    normalize=True,
    mu_c=2.5,
    sigma_pop_sd=0.5,
)

# Extract per-participant capability samples (compatible with suitability pipeline)
participant_idata = extract_population_capability_samples(
    idata,
    participant_names=list(perf_df.index),
    draws=2000,
)

# Visualize individual profiles
cap_df = collect_capability_means(participant_idata, model_capability_cols=capability_cols)
plot_radar_capabilities(cap_df, abilities_df, overlay=True, save_path="figures/population_radar.png")

# Load saved results in a later session
idata, participant_names, capability_cols = load_population_idata("data/results/population")
```

### Computing Suitability Scores

```python
from src.pipeline.suitability import score_all_tasks
from src.utils.io import load_ability_matrix, load_agent_idata

# Load ability matrix and agent inference data
ability_matrix = load_ability_matrix(
    "data/processed/ability_matrix_combined.csv",
    tasks_df, abilities_df, use_short_names=True
)
agent_idata, capability_cols = load_agent_idata(
    "data/results/simulated",
    ["strong_generalist", "weak_generalist"]
)

# Compute suitability scores
mean_df, ci_lo_df, ci_hi_df, samples = score_all_tasks(
    agent_idata=agent_idata,
    capability_cols=list(abilities_df["Abilities"]),
    demand_df=ability_matrix,
    weight_uncertainty="dirichlet",
)
```

## License

MIT License

## Citation

If you use this work, please cite:

```bibtex
@software{task_ability_profiles,
  title = {Task-Ability Profiles: A Capability-Based Profiling System},
  year = {2025},
  url = {https://github.com/jonnyp1990/task-ability-profiles}
}
```
