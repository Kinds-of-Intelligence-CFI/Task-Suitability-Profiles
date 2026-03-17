"""
Agent simulation module.

Simulates agent performance on tasks based on predefined capability profiles.
Used for validation and testing of the inference pipeline.
"""

import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from typing import Dict, List, Optional, Tuple


def create_agent_profiles(
    capability_cols: List[str],
    profiles: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Create capability profiles for simulated agents.

    Args:
        capability_cols: List of capability names
        profiles: Optional dict mapping agent names to capability dicts.
                  If None, creates default profiles.

    Returns:
        DataFrame with agents as rows, capabilities as columns
    """
    if profiles is None:
        # Default agent profiles
        profiles = {
            "strong_generalist": {name: 3.0 for name in capability_cols},
            "weak_generalist": {name: 1.5 for name in capability_cols},
            "social_specialist": {
                **{name: 1.0 for name in capability_cols},
                "Theory of Mind": 5.0,
                "Language": 5.0,
            },
            "strategic_specialist": {
                **{name: 1.0 for name in capability_cols},
                "Mental Simulation": 5.0,
                "Planning": 5.0,
            },
            "physical_specialist": {
                **{name: 1.0 for name in capability_cols},
                "Procedural Memory": 5.0,
                "Spatial Reasoning and Navigation": 5.0,
            },
        }

    C_df = pd.DataFrame.from_dict(profiles, orient="index")
    C_df = C_df.reindex(columns=capability_cols)
    return C_df


def simulate_agent_performance(
    C: np.ndarray,
    D: np.ndarray,
    lam: float = 1.0,
    kappa: Optional[np.ndarray] = None,
    alpha: float = 0.0,
    gamma0: Optional[float] = None,
    guess: Optional[float] = None,
    slip: Optional[float] = None,
    pool: str = "add",
    tau: float = 1.0,
    normalize: bool = False,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulate binary performance outcomes for agents on items.

    Model:
        For each agent i and item j:
        - margin[i,j,k] = c[i,k] - lam * D[j,k]  (if D[j,k] > 0)
        - z[i,j] depends on pooling method (see pool parameter)
        - p[i,j] = sigmoid(z[i,j])
        - Y[i,j] ~ Bernoulli(p[i,j])

    Args:
        C: Agent capability matrix of shape (I, K) where I is number of agents
        D: Item demand matrix of shape (J, K) where J is number of items
        lam: Per-level log step (how much each demand level reduces success prob)
        kappa: Discrimination weights of shape (K,). Default: all ones.
        alpha: Intercept (baseline log-odds when capabilities match demands)
        gamma0: Optional bonus to logits when an item has all demands == 0
        guess: Guessing parameter (lower asymptote)
        slip: Slip parameter (upper asymptote = 1 - slip)
        pool: Pooling method:
              - "add": Weighted sum (compensatory - capabilities can substitute)
              - "softmin": Soft-minimum (non-compensatory - weakest capability limits success)
        tau: Temperature for softmin pooling (higher = stricter weakest-link)
        normalize: If True, divide additive pooling by number of active capabilities
                   per item (mean instead of sum)
        seed: Random seed

    Returns:
        Tuple of:
        - Y: Binary outcomes of shape (I, J)
        - p: Success probabilities of shape (I, J)
        - z: Log-odds of shape (I, J)
    """
    rng = np.random.default_rng(seed)
    C = np.atleast_2d(np.asarray(C, float))
    D = np.asarray(D, float)
    I, Kc = C.shape
    J, Kd = D.shape
    assert Kc == Kd, f"C has {Kc} capabilities but D has {Kd}"
    K = Kc

    lam = np.broadcast_to(np.asarray(lam, float), (K,))
    if kappa is None:
        kappa = np.ones(K)
    else:
        kappa = np.broadcast_to(np.asarray(kappa, float), (K,))

    # Compute margins only where demand > 0
    is_on = (D > 0).astype("float64")
    # Shape: (I, J, K)
    margin = is_on[None, :, :] * (C[:, None, :] - lam[None, None, :] * D[None, :, :])
    weighted_margin = kappa[None, None, :] * margin

    # Aggregate across capabilities
    if pool == "add":
        z = alpha + np.sum(weighted_margin, axis=2)
        if normalize:
            K_on = is_on.sum(axis=1)  # (J,)
            K_on_safe = np.maximum(K_on, 1.0)
            z = alpha + np.sum(weighted_margin, axis=2) / K_on_safe[None, :]

    elif pool == "softmin":
        # Soft-minimum: z = -log(mean(exp(-tau * margin))) / tau
        # Weakest capability (lowest margin) dominates.
        # As tau -> inf, approaches hard minimum (strictly non-compensatory).
        # As tau -> 0, approaches arithmetic mean (identical to normadd).
        K_on = is_on.sum(axis=1)  # (J,)
        K_on_safe = np.maximum(K_on, 1.0)
        if tau == 0.0:
            # Limiting case: arithmetic mean of margins (= normadd)
            z_mean = alpha + np.sum(weighted_margin, axis=2) / K_on_safe[None, :]
            z = np.where(K_on[None, :] > 0, z_mean, alpha)
        else:
            # Clamp to prevent overflow
            wm_clamped = np.clip(weighted_margin, -20, 20)
            u = -tau * wm_clamped
            # Mask inactive capabilities with -inf so exp(-inf)=0
            u_masked = np.where(is_on[None, :, :] > 0, u, -np.inf)
            # logsumexp along capability axis
            u_max = np.max(u_masked, axis=2, keepdims=True)
            # Handle all-inactive items (u_max = -inf)
            u_max_safe = np.where(np.isfinite(u_max), u_max, 0.0)
            with np.errstate(divide="ignore"):
                lse = u_max_safe.squeeze(2) + np.log(
                    np.sum(np.exp(u_masked - u_max_safe), axis=2)
                )
            z_softmin = alpha - (lse - np.log(K_on_safe[None, :])) / tau
            z = np.where(K_on[None, :] > 0, z_softmin, alpha)

    else:
        raise ValueError(f'pool must be "add" or "softmin", got "{pool}"')

    # Optional boost for zero-demand items
    if gamma0 is not None:
        all_zero = (D == 0).all(axis=1).astype(float)
        z = z + gamma0 * all_zero[None, :]

    # Compute probabilities
    p_base = sigmoid(z)

    # Optional guessing/slip parameters
    if guess is not None or slip is not None:
        g = 0.0 if guess is None else float(guess)
        s = 0.0 if slip is None else float(slip)
        p = g + (1 - g - s) * p_base
    else:
        p = p_base

    # Sample outcomes
    Y = rng.binomial(1, p)

    return Y, p, z


def create_simulated_data(
    capability_cols: List[str],
    D: np.ndarray,
    agent_profiles: Optional[Dict[str, Dict[str, float]]] = None,
    lam: float = 1.0,
    pool: str = "add",
    tau: float = 1.0,
    normalize: bool = False,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create simulated performance data for a set of agents.

    Args:
        capability_cols: List of capability names
        D: Demand matrix of shape (J, K)
        agent_profiles: Optional dict of agent capability profiles
        lam: Per-level log step
        pool: Pooling method ("add" or "softmin")
        tau: Temperature for softmin pooling
        normalize: If True, use mean instead of sum for additive pooling
        seed: Random seed

    Returns:
        Tuple of:
        - C_df: Capability profiles DataFrame
        - Ym_df: Binary outcomes DataFrame (agents x items)
        - Pm_df: Success probabilities DataFrame (agents x items)
    """
    C_df = create_agent_profiles(capability_cols, agent_profiles)
    C = C_df.to_numpy(float)

    Y, p, _ = simulate_agent_performance(
        C=C,
        D=D,
        lam=lam,
        kappa=np.ones(len(capability_cols)),
        alpha=0.0,
        pool=pool,
        tau=tau,
        normalize=normalize,
        seed=seed,
    )

    item_index = np.arange(D.shape[0])
    Ym_df = pd.DataFrame(Y, index=C_df.index, columns=item_index)
    Pm_df = pd.DataFrame(p, index=C_df.index, columns=item_index)

    return C_df, Ym_df, Pm_df
