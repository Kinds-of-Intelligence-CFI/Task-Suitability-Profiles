"""
Inference pipeline module.

Runs Bayesian inference to estimate capability profiles from performance data.
"""

import numpy as np
import pandas as pd
import arviz as az
from typing import Dict, List, Optional, Tuple

from ..core.model import build_capability_model, fit_model, fit_agent, fit_population
from ..utils.io import save_agent_idata, save_model_data, save_population_idata


def run_inference(
    agent_name: str,
    performance_df: pd.DataFrame,
    demand_matrix: np.ndarray,
    item_index: np.ndarray,
    output_base_path: str,
    lam: float = 1.0,
    gamma0: Optional[float] = None,
    seed: int = 42,
    save_results: bool = True,
    hierarchical: bool = True,
    fix_kappa: bool = True,
    coverage_aware: bool = False,
    pool: str = "add",
    tau: float = 1.0,
    normalize: bool = False,
    mu_c: float = 3.0,
    capability_cols: Optional[List[str]] = None,
) -> Tuple[az.InferenceData, object]:
    """
    Run inference for a single agent.

    Args:
        agent_name: Name of the agent
        performance_df: DataFrame with agents as rows, items as columns
        demand_matrix: Array of shape (J, K) with item demands
        item_index: Item identifiers
        output_base_path: Base path for saving results
        lam: Per-level log step
        gamma0: Optional boost for zero-demand items
        seed: Random seed
        save_results: If True, save results to disk
        hierarchical: If True, use hierarchical prior (recommended)
        fix_kappa: If True, fix kappa=1 (recommended for identifiability)
        coverage_aware: If True, adjust priors based on coverage
        pool: Pooling method ("add", "geom", or "softmin")
        tau: Temperature for softmin pooling (higher = stricter weakest-link)
        normalize: If True, use mean instead of sum for additive pooling
        mu_c: Prior mean for capability levels (increase for softmin)
        capability_cols: List of capability column names in model parameter order.
                         Saved as metadata for correct visualization mapping.

    Returns:
        Tuple of (InferenceData, Model)
    """
    print(f"Fitting model for {agent_name}...")

    idata, model = fit_agent(
        agent_name=agent_name,
        performance_df=performance_df,
        demand_matrix=demand_matrix,
        item_index=item_index,
        lam=lam,
        gamma0=gamma0,
        seed=seed,
        hierarchical=hierarchical,
        fix_kappa=fix_kappa,
        coverage_aware=coverage_aware,
        pool=pool,
        tau=tau,
        normalize=normalize,
        mu_c=mu_c,
    )

    if save_results:
        save_agent_idata(idata, agent_name, output_base_path, capability_cols=capability_cols)

    return idata, model


def run_inference_batch(
    agents: List[str],
    performance_df: pd.DataFrame,
    demand_matrix: np.ndarray,
    item_index: np.ndarray,
    output_base_path: str,
    lam: float = 1.0,
    gamma0: Optional[float] = None,
    base_seed: int = 100,
    save_results: bool = True,
    hierarchical: bool = True,
    fix_kappa: bool = True,
    coverage_aware: bool = False,
    pool: str = "add",
    tau: float = 1.0,
    normalize: bool = False,
    mu_c: float = 3.0,
    capability_cols: Optional[List[str]] = None,
) -> Dict[str, Tuple[az.InferenceData, object]]:
    """
    Run inference for multiple agents sequentially.

    Args:
        agents: List of agent names to fit
        performance_df: DataFrame with agents as rows, items as columns
        demand_matrix: Array of shape (J, K) with item demands
        item_index: Item identifiers
        output_base_path: Base path for saving results
        lam: Per-level log step
        gamma0: Optional boost for zero-demand items
        base_seed: Base random seed (incremented for each agent)
        save_results: If True, save results to disk
        hierarchical: If True, use hierarchical prior (recommended)
        fix_kappa: If True, fix kappa=1 (recommended for identifiability)
        coverage_aware: If True, adjust priors based on coverage
        pool: Pooling method ("add", "geom", or "softmin")
        tau: Temperature for softmin pooling (higher = stricter weakest-link)
        normalize: If True, use mean instead of sum for additive pooling
        mu_c: Prior mean for capability levels (increase for softmin)
        capability_cols: List of capability column names in model parameter order.
                         Saved as metadata for correct visualization mapping.

    Returns:
        Dictionary mapping agent names to (InferenceData, Model) tuples
    """
    results = {}
    model_data = {}

    print(f"\n--- Starting inference for {len(agents)} agents ---")
    pool_str = pool + (" (normalized)" if normalize else "")
    if pool == "softmin":
        pool_str += f", tau={tau}"
    print(f"Model settings: hierarchical={hierarchical}, fix_kappa={fix_kappa}, coverage_aware={coverage_aware}, pool={pool_str}, mu_c={mu_c}")

    for i, agent in enumerate(agents, 1):
        print(f"\nFitting {i}/{len(agents)}: {agent}")

        idata, model = run_inference(
            agent_name=agent,
            performance_df=performance_df,
            demand_matrix=demand_matrix,
            item_index=item_index,
            output_base_path=output_base_path,
            lam=lam,
            gamma0=gamma0,
            seed=base_seed + i,
            save_results=save_results,
            hierarchical=hierarchical,
            fix_kappa=fix_kappa,
            coverage_aware=coverage_aware,
            pool=pool,
            tau=tau,
            normalize=normalize,
            mu_c=mu_c,
            capability_cols=capability_cols,
        )

        results[agent] = (idata, model)
        model_data[agent] = model

    if save_results:
        save_model_data(model_data, f"{output_base_path}_models.pkl")

    print("\n--- Inference complete ---")
    return results


def run_population_inference(
    performance_df: pd.DataFrame,
    demand_matrix: np.ndarray,
    item_index: np.ndarray,
    capability_cols: List[str],
    output_base_path: str,
    lam: float = 1.0,
    seed: int = 42,
    save_results: bool = True,
    pool: str = "add",
    tau: float = 1.0,
    normalize: bool = False,
    mu_c: float = 2.5,
    sigma_c: float = 0.8,
    sigma_pop_sd: float = 0.5,
    fix_kappa: bool = True,
) -> Tuple[az.InferenceData, object]:
    """
    Fit the hierarchical population IRT model across all participants.

    Args:
        performance_df: DataFrame with participants as rows, items as columns (0/1/NaN)
        demand_matrix: Array of shape (J, K) aligned with item_index
        item_index: Item identifiers matching performance_df columns
        capability_cols: List of K capability names in model parameter order
        output_base_path: Base path for saving results
        lam: Per-level log step
        seed: Random seed
        save_results: If True, save InferenceData and metadata
        pool: Pooling method ("add", "geom", or "softmin")
        tau: Temperature for softmin/geom pooling
        normalize: If True, use mean instead of sum for additive pooling
        mu_c: Prior mean for population capability levels
        sigma_c: Prior SD for population capability means
        sigma_pop_sd: Scale for HalfNormal prior on within-population spread
        fix_kappa: If True, fix kappa=1 for identifiability

    Returns:
        Tuple of (InferenceData, Model)
    """
    participants = list(performance_df.index)
    pool_str = pool + (" (normalized)" if normalize else "")
    if pool == "softmin":
        pool_str += f", tau={tau}"
    print(f"\n--- Population inference: {len(participants)} participants ---")
    print(f"  Pool: {pool_str} | mu_c={mu_c} | sigma_pop_sd={sigma_pop_sd} | fix_kappa={fix_kappa}")

    idata, model = fit_population(
        performance_df=performance_df,
        demand_matrix=demand_matrix,
        item_index=item_index,
        capability_cols=capability_cols,
        lam=lam,
        seed=seed,
        pool=pool,
        tau=tau,
        normalize=normalize,
        mu_c=mu_c,
        sigma_c=sigma_c,
        sigma_pop_sd=sigma_pop_sd,
        fix_kappa=fix_kappa,
    )

    if save_results:
        save_population_idata(
            idata,
            output_base_path,
            participant_names=participants,
            capability_cols=capability_cols,
        )

    print("--- Population inference complete ---")
    return idata, model


def summarize_inference(
    idata: az.InferenceData,
    var_names: List[str] = ["c", "theta", "kappa", "alpha"],
) -> pd.DataFrame:
    """
    Generate summary statistics for inference results.

    Args:
        idata: ArviZ InferenceData
        var_names: Variables to include in summary

    Returns:
        DataFrame with summary statistics
    """
    return az.summary(idata, var_names=var_names)


def diagnose_inference(
    idata: az.InferenceData,
) -> Dict[str, any]:
    """
    Run diagnostic checks on inference results.

    Args:
        idata: ArviZ InferenceData

    Returns:
        Dictionary with diagnostic results
    """
    diagnostics = {
        "r_hat": az.rhat(idata),
        "ess": az.ess(idata),
        "mcse": az.mcse(idata),
    }
    return diagnostics
