"""
Suitability scoring module.

Computes suitability scores for agents across different tasks based on
their inferred capability profiles and task demands.
"""

import numpy as np
import pandas as pd
import arviz as az
from typing import Dict, List, Optional, Tuple

from ..core.model import extract_capability_samples


def demand_weights(
    row: np.ndarray,
    normalize: bool = True,
    zero_as_zero: bool = True,
    sharpness: float = 1.0,
) -> np.ndarray:
    """
    Compute normalized weights from demand values.

    Args:
        row: Array of demand values
        normalize: If True, normalize to sum to 1
        zero_as_zero: If True, set near-zero values to exactly 0
        sharpness: Power to raise weights to (>1 emphasizes high demands)

    Returns:
        Array of weights
    """
    w = np.asarray(row, float).copy()
    if zero_as_zero:
        w[w < 1e-12] = 0.0

    if sharpness != 1.0:
        w = np.power(w, sharpness)

    if normalize:
        s = w.sum()
        if s > 0:
            w = w / s
    return w


def sample_weights_from_profile(
    row: np.ndarray,
    kappa: float = 200.0,
    sharpness: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Sample weights from Dirichlet distribution based on demand profile.

    Args:
        row: Array of demand values
        kappa: Concentration parameter (higher = more concentrated)
        sharpness: Power for demand weighting
        rng: Random number generator

    Returns:
        Array of sampled weights
    """
    rng = np.random.default_rng(rng)
    base = demand_weights(row, normalize=True, zero_as_zero=True, sharpness=sharpness)
    alpha = np.maximum(base * kappa, 1e-6)
    return rng.dirichlet(alpha)


def power_mean(
    values: np.ndarray,
    weights: np.ndarray,
    p: float = 2.0,
    eps: float = 1e-10,
) -> np.ndarray:
    """
    Compute weighted power mean.

    Args:
        values: Array of values, shape (n_samples, n_capabilities)
        weights: Array of weights, shape (n_samples, n_capabilities)
        p: Power parameter (p=1: arithmetic, p=0: geometric, p=2: quadratic)
        eps: Small value to avoid numerical issues

    Returns:
        Array of power means, shape (n_samples,)
    """
    values_pos = np.maximum(values, eps)

    if p == 1.0:
        return (weights * values_pos).sum(axis=1)
    elif p == 0:
        log_vals = np.log(values_pos)
        return np.exp((weights * log_vals).sum(axis=1))
    else:
        powered = np.power(values_pos, p)
        weighted_sum = (weights * powered).sum(axis=1)
        return np.power(weighted_sum, 1.0 / p)


def compute_suitability_scores(
    agent_idata: Dict[str, az.InferenceData],
    capability_cols: List[str],
    demand_df: pd.DataFrame,
    task: str,
    draws_cap: int = 2000,
    use_ratio: bool = True,
    weight_uncertainty: Optional[str] = None,
    kappa: float = 200.0,
    power_param: float = 2.0,
    demand_sharpness: float = 1.0,
    seed: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute suitability scores for a single task across all agents.

    Args:
        agent_idata: Dictionary mapping agent names to InferenceData
        capability_cols: List of capability column names
        demand_df: DataFrame with tasks as rows, capabilities as columns
        task: Task name (row in demand_df)
        draws_cap: Number of posterior draws to use
        use_ratio: If True (default), exponentiate capabilities (use theta instead of c)
        weight_uncertainty: If "dirichlet", sample weights from Dirichlet
        kappa: Concentration parameter for Dirichlet sampling
        power_param: Power for aggregation (1=arithmetic, 2=quadratic mean)
        demand_sharpness: Sharpness for demand weighting
        seed: Random seed

    Returns:
        Tuple of (samples DataFrame, summary DataFrame)
    """
    rng = np.random.default_rng(seed)

    w_row = demand_df.loc[task, capability_cols].to_numpy(float)
    fixed_w = demand_weights(
        w_row, normalize=True, zero_as_zero=True, sharpness=demand_sharpness
    )

    results = {}
    for agent, idata in agent_idata.items():
        cap_samples = extract_capability_samples(idata, var="c", draws=draws_cap)
        if use_ratio:
            cap_samples = np.exp(cap_samples)

        if weight_uncertainty == "dirichlet":
            W = np.vstack([
                sample_weights_from_profile(
                    w_row, kappa=kappa, sharpness=demand_sharpness, rng=rng
                )
                for _ in range(cap_samples.shape[0])
            ])
        else:
            W = np.broadcast_to(fixed_w, cap_samples.shape)

        S = power_mean(cap_samples, W, p=power_param)
        results[agent] = S

    S_df = pd.DataFrame(results)
    summary = pd.DataFrame({
        "mean": S_df.mean(axis=0),
        "sd": S_df.std(axis=0),
        "hdi_2.5%": S_df.quantile(0.025, axis=0),
        "hdi_97.5%": S_df.quantile(0.975, axis=0),
    }).sort_values("mean", ascending=False)

    return S_df, summary


def column_normalize_demands(
    demand_df: pd.DataFrame,
    capability_cols: List[str],
) -> pd.DataFrame:
    """
    Min-max normalise each capability column across tasks to [0, 1].

    This converts absolute demand levels into relative demand — how much does
    this task emphasise this ability compared to other tasks? Abilities that
    vary little across tasks (e.g. EM: 15-20) shrink toward uniform weights;
    abilities that vary a lot (e.g. EPaE: 0-28) retain their discriminating
    power. This prevents near-uniform absolute demand levels from washing out
    task-specific demand patterns.

    Args:
        demand_df: DataFrame with tasks as rows, capabilities as columns
        capability_cols: Columns to normalise (others left unchanged)

    Returns:
        Copy of demand_df with capability columns normalised to [0, 1]
    """
    demand_df = demand_df.copy()
    col_min = demand_df[capability_cols].min(axis=0)
    col_max = demand_df[capability_cols].max(axis=0)
    col_range = (col_max - col_min).replace(0, 1)  # avoid div-by-zero for constant columns
    demand_df[capability_cols] = (demand_df[capability_cols] - col_min) / col_range
    return demand_df


def score_all_tasks(
    agent_idata: Dict[str, az.InferenceData],
    capability_cols: List[str],
    demand_df: pd.DataFrame,
    tasks: Optional[List[str]] = None,
    draws_cap: int = 2000,
    use_ratio: bool = True,
    weight_uncertainty: str = "dirichlet",
    kappa: float = 300.0,
    power_param: float = 2.0,
    demand_sharpness: float = 1.0,
    column_normalize: bool = False,
    seed: int = 123,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Compute suitability scores for all tasks.

    Args:
        agent_idata: Dictionary mapping agent names to InferenceData
        capability_cols: List of capability column names
        demand_df: DataFrame with tasks as rows, capabilities as columns
        tasks: List of tasks to score (default: all rows in demand_df)
        draws_cap: Number of posterior draws to use
        use_ratio: If True (default), use exponentiated capabilities
        weight_uncertainty: Uncertainty method for weights
        kappa: Concentration parameter for Dirichlet
        power_param: Power for aggregation
        demand_sharpness: Sharpness for demand weighting
        column_normalize: If True, min-max normalise each capability column across
                          tasks before computing weights. Converts absolute demand
                          levels into relative demand (how much does this task
                          emphasise this ability vs other tasks). Improves score
                          variation across tasks when the raw demand matrix is
                          near-uniform.
        seed: Random seed

    Returns:
        Tuple of:
        - mean_df: Mean scores (tasks x agents)
        - ci_lo_df: Lower CI bounds
        - ci_hi_df: Upper CI bounds
        - samples_dict: Dictionary mapping tasks to sample DataFrames
    """
    if tasks is None:
        tasks = list(demand_df.index)

    # Validate columns
    missing = [c for c in capability_cols if c not in demand_df.columns]
    if missing:
        raise ValueError(f"Capability columns missing from demand_df: {missing}")

    if column_normalize:
        demand_df = column_normalize_demands(demand_df, capability_cols)

    mean_rows = []
    lo_rows = []
    hi_rows = []
    samples_dict = {}

    for task in tasks:
        S_samples, S_summary = compute_suitability_scores(
            agent_idata=agent_idata,
            capability_cols=capability_cols,
            demand_df=demand_df,
            task=task,
            draws_cap=draws_cap,
            use_ratio=use_ratio,
            weight_uncertainty=weight_uncertainty,
            kappa=kappa,
            power_param=power_param,
            demand_sharpness=demand_sharpness,
            seed=seed,
        )

        samples_dict[task] = S_samples
        mean_rows.append(pd.Series(S_summary["mean"], name=task))
        lo_rows.append(pd.Series(S_summary["hdi_2.5%"], name=task))
        hi_rows.append(pd.Series(S_summary["hdi_97.5%"], name=task))

    mean_df = pd.DataFrame(mean_rows).reindex(tasks)
    ci_lo_df = pd.DataFrame(lo_rows).reindex(tasks)
    ci_hi_df = pd.DataFrame(hi_rows).reindex(tasks)

    return mean_df, ci_lo_df, ci_hi_df, samples_dict
