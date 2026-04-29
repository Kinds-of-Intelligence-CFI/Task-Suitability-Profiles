# Suitability

This folder contains everything needed to infer capability profiles for AI models from benchmark performance data and compute suitability scores against workforce task demands.

## Overview

The suitability pipeline has three main stages:

1. **Build an ability matrix** — quantify the cognitive demands of each task or role using questionnaire data from workers
2. **Infer capability profiles** — fit a Bayesian IRT model to the model's benchmark performance to estimate its capability levels across 18 cognitive dimensions
3. **Compute suitability scores** — compare capability profiles to task demand profiles to produce interpretable suitability scores

Pre-computed outputs are provided at each stage so you can run any step independently.

## Quick Start with Pre-computed Data

To go straight to suitability scoring using pre-computed profiles and ability matrices:

```bash
python -m Suitability.scripts.run_suitability \
    --agents gpt-4o-mini \
    --idata-base Suitability/data/results/llm_normadd \
    --ability-matrix Suitability/data/processed/ability_matrix_combined.csv
```

Results are saved to `Suitability/data/results/` by default.

## 1. Build an Ability Matrix from Questionnaire Data

An ability matrix encodes how important each of the 18 cognitive capabilities is for each workforce task or role, derived from worker survey responses.

Pre-built matrices for several workforce domains are provided at `Suitability/data/processed/ability_matrix_<domain>.csv`. To build your own from questionnaire data:

```bash
# All participants, all domains
python -m Suitability.scripts.build_ability_matrix \
    --companies Suitability/data/raw/company_data.csv \
    --online Suitability/data/raw/online_data.csv \
    --output Suitability/data/processed/ability_matrix_combined.csv

# Specific domain only (e.g. domain 3 = NDP: Numerical, Data, or Programming)
python -m Suitability.scripts.build_ability_matrix \
    --companies Suitability/data/raw/company_data.csv \
    --online Suitability/data/raw/online_data.csv \
    --domain 3 \
    --output Suitability/data/processed/ability_matrix_NDP.csv

# Online participants only
python -m Suitability.scripts.build_ability_matrix \
    --online Suitability/data/raw/online_data.csv \
    --source online \
    --output Suitability/data/processed/ability_matrix_online.csv

# Build matrices for all domains in one run
python -m Suitability.scripts.build_ability_matrix \
    --companies Suitability/data/raw/company_data.csv \
    --online Suitability/data/raw/online_data.csv \
    --all-domains \
    --output Suitability/data/processed/ability_matrix
```

The questionnaire template is at `Suitability/data/raw/questionaire_template.csv`. The `--companies` and `--online` flags correspond to data collected directly from company employees and from an online panel respectively; both are optional and can be used in combination.

### Work Domains

| # | Type | Sub-type | Acronym |
|---|------|----------|---------|
| 1 | Manual-physical | Warehouse or logistics | WL |
| 2 | Manual-physical | Manufacture, maintenance, or repair | MMR |
| 3 | Computer-digital | Numerical, data, or programming | NDP |
| 4 | Computer-office | Admin, organisational, or planning | AOP |
| 5 | Computer-office | Customer service, marketing or HR | CMH |
| 6 | Face-to-face | Hospitality, sales, or client care | HSC |

## 2. Infer Capability Profiles

Given benchmark performance data and item annotations, fit the Bayesian IRT model to estimate capability levels for an AI model:

```bash
python -m Suitability.scripts.run_inference \
    --mode llm \
    --results Suitability/data/raw/gpt-4o-mini_results.csv \
    --annotations Suitability/data/processed/annotations.csv \
    --agent-name gpt-4o-mini \
    --output Suitability/data/results/llm_normadd
```

Pre-computed results for `gpt-4o-mini` are in `Suitability/data/results/` and can be used without re-running inference.

To infer profiles for a subset of capabilities:

```bash
python -m Suitability.scripts.run_inference \
    --mode llm \
    --results Suitability/data/raw/gpt-4o-mini_results.csv \
    --annotations Suitability/data/processed/annotations.csv \
    --agent-name gpt-4o-mini \
    --capabilities EM SM WM AaIC L MC \
    --output Suitability/data/results/llm_subset
```

Inference can take several minutes. The output is an ArviZ InferenceData file saved to the specified output folder.

## 3. Compute Suitability Scores

Match inferred capability profiles against task demand profiles:

```bash
# Recommended settings
python -m Suitability.scripts.run_suitability \
    --agents gpt-4o-mini \
    --idata-base Suitability/data/results/llm_normadd \
    --ability-matrix Suitability/data/processed/ability_matrix_combined.csv \
    --column-normalize --power 0.5 --use-ratio

# Domain-specific matrix
python -m Suitability.scripts.run_suitability \
    --agents gpt-4o-mini \
    --idata-base Suitability/data/results/llm_normadd \
    --ability-matrix Suitability/data/processed/ability_matrix_combined_domain_NDP.csv \
    --column-normalize --power 0.5 --use-ratio
```

Results are saved to `Suitability/data/results/` by default.

**Key options:**

- **`--column-normalize`** (recommended): min-max normalises each capability column across tasks before computing demand weights. Raw ability matrices tend to have near-uniform demand levels, which collapses scores toward the same value regardless of the agent's profile. Column normalisation converts absolute demand into *relative* demand — how much does this task emphasise this capability compared to others — making scores sensitive to the match between the agent's capabilities and each task's specific demands.

- **`--power`**: Controls compensability of the weighted power mean. `p=0.5` (square-root mean) is empirically robust for importance-weighted ability matrices. `p=1` is arithmetic mean; `p=0` is geometric mean.

- **`--use-ratio`**: Uses ratio-scale capabilities `theta = exp(c)` instead of log-scale `c`. Recommended when `p=0.5`.

> **Note on cross-matrix comparability**: column normalisation is fitted separately to each matrix. Scores from different domain-specific matrices are not directly comparable on the same scale.

## 4. Visualise Capability Profiles

```bash
python -m Suitability.scripts.visualize_profiles \
    --agents gpt-4o-mini \
    --idata-base Suitability/data/results/llm_normadd \
    --output figures/
```

Add `--plot-forest` to include forest plots of posterior distributions in addition to radar plots.

## The Capability Framework

### 18 Cognitive Capabilities

| # | Capability | Acronym | Description |
|---|-----------|---------|-------------|
| 1 | Episodic Memory | EM | Remembering previous events |
| 2 | Semantic Memory | SM | Remembering facts and information |
| 3 | Procedural Memory | ProcM | Remembering how to perform learned tasks |
| 4 | Prospective Memory | ProsM | Remembering to do planned actions |
| 5 | Working Memory | WM | Holding multiple pieces of information at once |
| 6 | Attention and Inhibitory Control | AaIC | Controlling behaviours to focus on tasks |
| 7 | Cognitive Flexibility | CF | Switching between tasks or adapting |
| 8 | Mental Simulation | MS | Imagining possible future scenarios |
| 9 | Planning | P | Mapping out strategies to achieve goals |
| 10 | Metacognition | MC | Assessing own thoughts and performance |
| 11 | Perception and Pattern Recognition | PaPR | Noticing relevant details |
| 12 | Functional Perception | FP | Recognising appropriate roles for objects or people |
| 13 | Spatial Reasoning and Navigation | SRaN | Reasoning about size, space, and distance |
| 14 | Object Permanence | OP | Understanding objects exist when not visible |
| 15 | Causal Reasoning | CR | Understanding cause and effect |
| 16 | Theory of Mind | ToM | Reasoning about others' goals and beliefs |
| 17 | Emotion Perception and Empathy | EPaE | Connecting with others' feelings |
| 18 | Language | L | Using language to communicate |

## The Bayesian Model

The model uses Measurement Layouts ([arXiv:2309.11975](https://arxiv.org/abs/2309.11975)) to infer latent capability levels from binary performance data.

For agent *i* on item *j* requiring capability *k*:

```
c[k] ~ Normal(mu_c, sigma_c)           # Log-scale capability
delta[j,k] = exp(lambda * D[j,k])     # Difficulty for item j, capability k
margin[j,k] = c[k] - lambda * D[j,k]  # Log-ratio of capability to difficulty
z[j] = alpha + mean_k(margin[j,k])    # Log-odds (normalized additive pooling)
p[j] = sigmoid(z[j])                  # Success probability
Y[j] ~ Bernoulli(p[j])                # Observed outcome
```

`D[j,k]` is the demand level (0–5) for capability *k* on item *j*, as annotated by the LLM annotator. A positive margin means the agent's capability exceeds the item's demand.

### Pooling Methods

| Method | Behaviour | Recommendation |
|--------|-----------|---------------|
| `add` + `normalize` | Compensatory — capabilities substitute for each other; scale-invariant | **Recommended default** |
| `softmin` | Non-compensatory — weakest capability limits success | Use when a bottleneck is theoretically motivated |
| `add` | Compensatory but logit scale grows with number of active capabilities | Avoid for multi-capability items |

## Suitability Scoring

Suitability scores combine inferred capabilities with task demand profiles using a weighted power mean:

```
S_task = (sum_k w[k] * theta[k]^p)^(1/p)
```

Where `w[k]` are demand weights from the ability matrix, `theta[k] = exp(c[k])` are posterior capability samples, and `p` is the power parameter. Uncertainty is propagated via posterior sampling over both capabilities and (optionally) demand weights.
