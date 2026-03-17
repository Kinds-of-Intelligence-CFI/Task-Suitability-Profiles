"""
Bayesian capability model building and fitting.

This module implements the core Item Response Theory (IRT) model for inferring
agent capability levels from observed performance data.
"""

import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
import arviz as az
from typing import Optional, Tuple, Literal, Union


def compute_capability_coverage(D: np.ndarray) -> np.ndarray:
    """
    Compute the coverage (proportion of non-zero demands) for each capability.

    Args:
        D: Demand matrix of shape (J, K)

    Returns:
        Array of shape (K,) with coverage proportions
    """
    return (D > 0).mean(axis=0)


def build_capability_model(
    Y: np.ndarray,
    D: np.ndarray,
    *,
    lam: float = 1.0,
    mu_c: float = 2.5,
    sigma_c: float = 0.8,
    alpha_prior: Tuple[float, float] = (0.0, 1.0),
    kappa_sd: float = 1.0,
    gamma0: Optional[float] = None,
    pool: Literal["add", "geom", "softmin"] = "add",
    tau: float = 1.0,
    normalize: bool = False,
    hierarchical: bool = False,
    fix_kappa: bool = False,
    coverage_aware: bool = False,
) -> pm.Model:
    """
    Build a Bayesian capability model for a single agent.

    The model estimates latent capability levels (c) from binary performance
    data (Y) given item demand levels (D).

    Model structure:
        - c[k] ~ Normal(mu_c, sigma_c)     # log-scale capability: c[k] = log(theta[k])
        - theta[k] = exp(c[k])             # ratio-scale capability (stored as deterministic)
        - kappa[k] ~ HalfNormal(kappa_sd)  # discrimination weight for ability k
        - alpha ~ Normal(alpha_prior)       # intercept (baseline log-odds)

        For each item j and capability k:
        - delta[j,k] = exp(lam * D[j,k])            # ratio-scale difficulty
        - margin[j,k] = c[k] - lam * D[j,k]         # log(theta[k] / delta[j,k])
                       = 0  if D[j,k] == 0 (capability not required)

        The margin is a log-ratio of capability to difficulty on the ratio scale.
        It is negative when theta[k] < delta[j,k] (capability falls short of demand)
        and positive when theta[k] > delta[j,k] (capability exceeds demand).

        - z[j] = alpha + sum_k(kappa[k] * margin[j,k])  (additive pooling)
        - p[j] = sigmoid(z[j])
        - Y[j] ~ Bernoulli(p[j])

        Note: inference is done entirely in log-space (on c[k]); theta[k] = exp(c[k])
        is the conceptually primary quantity but is only used for reporting.

    Args:
        Y: Binary performance vector of shape (J,) where J is number of items
        D: Demand matrix of shape (J, K) where K is number of capabilities
        lam: Per-level log step; each +1 demand step multiplies raw difficulty by exp(lam),
             i.e. delta[j,k] = exp(lam * D[j,k])
        mu_c: Prior mean for capability levels
        sigma_c: Prior standard deviation for capability levels
        alpha_prior: Tuple of (mean, sd) for the intercept prior
        kappa_sd: Standard deviation for HalfNormal prior on discrimination weights
        gamma0: Optional fixed boost to logits when all demands are 0 for an item
        pool: Pooling method:
              - "add": Weighted sum (compensatory - capabilities can substitute)
              - "geom": Log-mean-exp (soft-max, dominated by strongest capabilities)
              - "softmin": Soft-minimum (non-compensatory - weakest capability limits success)
        tau: Temperature parameter. For "softmin", higher tau = stricter weakest-link constraint.
        normalize: If True, divide additive pooling by number of active capabilities per item
                   (mean instead of sum). Makes logit scale independent of number of active
                   capabilities, improving stability across capability subsets.
        hierarchical: If True, use hierarchical prior for capabilities (shares info across capabilities)
        fix_kappa: If True, fix kappa=1 for all capabilities (improves identifiability)
        coverage_aware: If True, adjust prior uncertainty based on data coverage

    Returns:
        PyMC Model object ready for sampling
    """
    Y = np.asarray(Y).astype("int8")
    D = np.asarray(D, dtype=float)
    J, K = D.shape

    lam_arr = np.full(K, float(lam)) if np.isscalar(lam) else np.asarray(lam, float)
    kappa_sd_arr = np.full(K, float(kappa_sd)) if np.isscalar(kappa_sd) else np.asarray(kappa_sd, float)

    # Compute coverage for coverage-aware priors
    coverage = compute_capability_coverage(D)

    with pm.Model() as model:
        # Capability priors
        if hierarchical:
            # Hierarchical prior: share information across capabilities
            mu_pop = pm.Normal("mu_pop", mu=mu_c, sigma=1.0)
            sigma_pop = pm.HalfNormal("sigma_pop", sigma=0.5)
            c_offset = pm.Normal("c_offset", mu=0, sigma=1, shape=K)
            c = pm.Deterministic("c", mu_pop + sigma_pop * c_offset)
        elif coverage_aware:
            # Coverage-aware prior: tighter priors for sparse capabilities
            # Scale sigma inversely with sqrt(coverage) - sparse = tighter prior
            min_coverage = 0.1  # floor to avoid division issues
            coverage_factor = np.sqrt(np.maximum(coverage, min_coverage))
            sigma_k = sigma_c * coverage_factor  # smaller sigma for sparse capabilities
            c = pm.Normal("c", mu=mu_c, sigma=sigma_k, shape=K)
        else:
            # Standard independent prior
            c = pm.Normal("c", mu=mu_c, sigma=sigma_c, shape=K)

        theta = pm.Deterministic("theta", pm.math.exp(c))

        # Discrimination weights
        if fix_kappa:
            # Fix kappa=1 for identifiability
            kappa = pt.ones(K)
        else:
            kappa = pm.HalfNormal("kappa", sigma=kappa_sd_arr, shape=K)

        alpha = pm.Normal("alpha", mu=alpha_prior[0], sigma=alpha_prior[1])

        # Data
        d = pm.Data("d", D)
        is_on = pt.gt(d, 0).astype("float64")
        margin = is_on * (c[None, :] - lam_arr[None, :] * d)

        # Pooling
        if pool == "add":
            contrib = kappa[None, :] * margin
            z_sum = pt.sum(contrib, axis=1)
            if normalize:
                K_on = pt.sum(is_on, axis=1)
                K_on_safe = pm.math.maximum(K_on, 1.0)
                z_core = alpha + z_sum / K_on_safe
            else:
                z_core = alpha + z_sum

        elif pool == "geom":
            u = tau * (kappa[None, :] * margin)
            u_masked = pt.switch(is_on > 0, u, -np.inf)
            K_on = pt.sum(is_on, axis=1)
            K_on_safe = pm.math.maximum(K_on, 1.0)
            z_core = alpha + (pm.math.logsumexp(u_masked, axis=1) - pt.log(K_on_safe)) / tau

        elif pool == "softmin":
            # Soft-minimum: z = -log(mean(exp(-tau * margin))) / tau
            # This makes the weakest capability (lowest margin) dominate.
            # As tau -> inf, approaches true minimum (strictest weakest-link).
            # As tau -> 0, approaches arithmetic mean (identical to normadd).
            weighted_margin = kappa[None, :] * margin
            K_on = pt.sum(is_on, axis=1)
            K_on_safe = pm.math.maximum(K_on, 1.0)
            if tau == 0.0:
                # Limiting case: arithmetic mean of margins (= normadd)
                z_mean = alpha + pt.sum(weighted_margin, axis=1) / K_on_safe
                z_core = pt.switch(K_on > 0, z_mean, alpha)
            else:
                # Clamp margins to prevent numerical overflow in exp
                weighted_margin_clamped = pt.clip(weighted_margin, -20, 20)
                u = -tau * weighted_margin_clamped
                # For inactive capabilities, use -inf so exp(-inf)=0 doesn't affect logsumexp
                u_masked = pt.switch(is_on > 0, u, -np.inf)
                # softmin = -log(mean(exp(-tau*x)))/tau = -(logsumexp(-tau*x) - log(K))/tau
                lse = pm.math.logsumexp(u_masked, axis=1)
                # Handle items with no active capabilities (all demands = 0)
                # In this case logsumexp returns -inf; set z_core to alpha (baseline)
                z_softmin = alpha - (lse - pt.log(K_on_safe)) / tau
                z_core = pt.switch(K_on > 0, z_softmin, alpha)

        else:
            raise ValueError('pool must be "add", "geom", or "softmin"')

        # Optional boost for zero-demand items
        if gamma0 is not None:
            all_zero = pt.eq(pt.sum(pt.eq(d, 0), axis=1), K).astype("float64")
            z_core = z_core + gamma0 * all_zero

        # Likelihood
        z = pm.Deterministic("z_logit", z_core)
        p = pm.Deterministic("p", pm.math.sigmoid(z))
        pm.Bernoulli("y", logit_p=z, observed=Y)

    return model


def fit_model(
    model: pm.Model,
    draws: int = 4000,
    tune: int = 1500,
    target_accept: float = 0.95,
    seed: int = 42,
    sampler: str = "nutpie",
) -> az.InferenceData:
    """
    Fit a capability model using MCMC sampling.

    Args:
        model: PyMC Model to fit
        draws: Number of posterior draws per chain
        tune: Number of tuning steps
        target_accept: Target acceptance rate for NUTS sampler
        seed: Random seed for reproducibility
        sampler: NUTS sampler backend ("nutpie", "pymc", "blackjax")

    Returns:
        ArviZ InferenceData object containing posterior samples
    """
    with model:
        idata = pm.sample(
            tune=tune,
            draws=draws,
            target_accept=target_accept,
            random_seed=seed,
            nuts_sampler=sampler,
        )
    return idata


def fit_agent(
    agent_name: str,
    performance_df: pd.DataFrame,
    demand_matrix: np.ndarray,
    item_index: np.ndarray,
    lam: float = 1.0,
    gamma0: Optional[float] = None,
    seed: int = 42,
    hierarchical: bool = True,
    fix_kappa: bool = True,
    coverage_aware: bool = False,
    pool: Literal["add", "geom", "softmin"] = "add",
    tau: float = 1.0,
    normalize: bool = False,
    **model_kwargs,
) -> Tuple[az.InferenceData, pm.Model]:
    """
    Fit a capability model for a single agent.

    Args:
        agent_name: Name of the agent (row in performance_df)
        performance_df: DataFrame with agents as rows, items as columns, values are 0/1
        demand_matrix: Array of shape (J, K) with item demands
        item_index: Index/column names for items in performance_df
        lam: Per-level log step
        gamma0: Optional fixed boost for zero-demand items
        seed: Random seed
        hierarchical: If True, use hierarchical prior (recommended for better estimates)
        fix_kappa: If True, fix kappa=1 for identifiability (recommended)
        coverage_aware: If True, adjust priors based on capability coverage
        pool: Pooling method ("add", "geom", or "softmin")
        tau: Temperature for softmin pooling (higher = stricter weakest-link)
        normalize: If True, use mean instead of sum for additive pooling
        **model_kwargs: Additional arguments passed to build_capability_model

    Returns:
        Tuple of (InferenceData, Model)
    """
    Y = performance_df.loc[agent_name, item_index].to_numpy(int)

    model = build_capability_model(
        Y=Y,
        D=demand_matrix,
        lam=lam,
        mu_c=model_kwargs.get("mu_c", 3.0),
        sigma_c=model_kwargs.get("sigma_c", 1.0),
        alpha_prior=model_kwargs.get("alpha_prior", (0.0, 0.5)),
        kappa_sd=model_kwargs.get("kappa_sd", 1.0),
        gamma0=gamma0,
        pool=pool,
        tau=tau,
        normalize=normalize,
        hierarchical=hierarchical,
        fix_kappa=fix_kappa,
        coverage_aware=coverage_aware,
    )

    idata = fit_model(model, seed=seed)
    return idata, model


def build_population_model(
    Y_matrix: np.ndarray,
    D: np.ndarray,
    participant_names: list,
    capability_cols: list,
    *,
    lam: float = 1.0,
    mu_c: float = 2.5,
    sigma_c: float = 0.8,
    sigma_pop_sd: float = 0.5,
    alpha_prior: Tuple[float, float] = (0.0, 1.0),
    pool: Literal["add", "geom", "softmin"] = "add",
    tau: float = 1.0,
    normalize: bool = False,
    fix_kappa: bool = True,
) -> pm.Model:
    """
    Build a hierarchical population IRT model for multiple participants.

    Each participant has their own capability vector drawn from a shared population
    distribution (partial pooling). This is appropriate when participants complete
    a short battery where individual posteriors would be too wide without borrowing
    strength across individuals.

    Model structure:
        Population level:
            mu_pop[k]    ~ Normal(mu_c, sigma_c)      # population mean capability
            sigma_pop[k] ~ HalfNormal(sigma_pop_sd)   # population spread

        Individual level (non-centered):
            c_offset[n,k] ~ Normal(0, 1)
            c[n,k] = mu_pop[k] + sigma_pop[k] * c_offset[n,k]
            theta[n,k] = exp(c[n,k])                  # ratio-scale (for reporting)

        Item level (same pooling methods as build_capability_model):
            margin[n,j,k] = c[n,k] - lam * D[j,k]   (0 where D[j,k]==0)
            z[n,j]  = alpha + pooling(margin[n,j,:])
            Y[n,j] ~ Bernoulli(sigmoid(z[n,j]))

        Missing responses (NaN in Y_matrix) are excluded from the likelihood.

    Args:
        Y_matrix: Binary performance matrix of shape (N, J). May contain NaN
                  for items not completed by a participant.
        D: Demand matrix of shape (J, K)
        participant_names: List of N participant identifiers
        capability_cols: List of K capability names (for ArviZ coordinates)
        lam: Per-level log step
        mu_c: Prior mean for population capability levels
        sigma_c: Prior SD for population capability means
        sigma_pop_sd: Scale for HalfNormal prior on within-population spread
        alpha_prior: (mean, sd) for the shared intercept prior
        pool: Pooling method ("add", "geom", or "softmin")
        tau: Temperature for softmin/geom pooling
        normalize: If True, divide additive sum by number of active capabilities
        fix_kappa: If True, fix kappa=1 (recommended for identifiability)

    Returns:
        PyMC Model object ready for sampling
    """
    Y_matrix = np.asarray(Y_matrix, dtype=float)
    D = np.asarray(D, dtype=float)
    N, J = Y_matrix.shape
    K = D.shape[1]

    lam_arr = np.full(K, float(lam)) if np.isscalar(lam) else np.asarray(lam, float)
    lam_D = lam_arr[None, :] * D          # (J, K) - constant
    is_on = (D > 0).astype(float)         # (J, K) - constant
    K_on = is_on.sum(axis=-1)             # (J,)   - constant
    K_on_safe = np.maximum(K_on, 1.0)    # (J,)   - constant

    # Missing data: flatten to observed (participant, item) pairs
    obs_mask = ~np.isnan(Y_matrix)        # (N, J)
    obs_n, obs_j = np.where(obs_mask)
    Y_obs = Y_matrix[obs_n, obs_j].astype("int8")

    coords = {"participant": participant_names, "capability": capability_cols}

    with pm.Model(coords=coords) as model:
        # --- Population hyperpriors ---
        mu_pop = pm.Normal("mu_pop", mu=mu_c, sigma=sigma_c, dims="capability")
        sigma_pop = pm.HalfNormal("sigma_pop", sigma=sigma_pop_sd, dims="capability")

        # --- Individual capabilities (non-centered) ---
        c_offset = pm.Normal(
            "c_offset", mu=0.0, sigma=1.0, dims=("participant", "capability")
        )
        c = pm.Deterministic(
            "c", mu_pop[None, :] + sigma_pop[None, :] * c_offset,
            dims=("participant", "capability"),
        )
        theta = pm.Deterministic("theta", pm.math.exp(c), dims=("participant", "capability"))

        # Discrimination weights (shared across participants)
        if fix_kappa:
            kappa = pt.ones(K)
        else:
            kappa = pm.HalfNormal("kappa", sigma=1.0, dims="capability")

        alpha = pm.Normal("alpha", mu=alpha_prior[0], sigma=alpha_prior[1])

        # --- Margin computation: (N, J, K) ---
        # c[:, None, :] is (N, 1, K); lam_D[None, :, :] is (1, J, K)
        margin = c[:, None, :] - lam_D[None, :, :]          # (N, J, K)
        margin_masked = margin * is_on[None, :, :]            # zero inactive capabilities

        # --- Pooling ---
        if pool == "add":
            contrib = kappa[None, None, :] * margin_masked    # (N, J, K)
            z_sum = pt.sum(contrib, axis=-1)                  # (N, J)
            if normalize:
                z_core = alpha + z_sum / K_on_safe[None, :]
            else:
                z_core = alpha + z_sum

        elif pool == "geom":
            u = tau * kappa[None, None, :] * margin_masked    # (N, J, K)
            u_masked = pt.switch(
                pt.gt(is_on[None, :, :], 0), u,
                pt.zeros_like(u) - np.inf,
            )
            lse = pm.math.logsumexp(u_masked, axis=-1)        # (N, J)
            z_core = alpha + (lse - pt.log(K_on_safe)[None, :]) / tau

        elif pool == "softmin":
            wm = kappa[None, None, :] * margin_masked         # (N, J, K)
            wm_clamped = pt.clip(wm, -20, 20)
            u = -tau * wm_clamped
            u_masked = pt.switch(
                pt.gt(is_on[None, :, :], 0), u,
                pt.zeros_like(u) - np.inf,
            )
            lse = pm.math.logsumexp(u_masked, axis=-1)        # (N, J)
            z_softmin = alpha - (lse - pt.log(K_on_safe)[None, :]) / tau
            has_active = pt.gt(K_on, 0)[None, :]              # (1, J)
            z_core = pt.switch(has_active, z_softmin, alpha)

        else:
            raise ValueError('pool must be "add", "geom", or "softmin"')

        # --- Likelihood over observed responses only ---
        z_obs_vals = z_core[obs_n, obs_j]
        pm.Bernoulli("y", logit_p=z_obs_vals, observed=Y_obs)

    return model


def fit_population(
    performance_df: pd.DataFrame,
    demand_matrix: np.ndarray,
    item_index: np.ndarray,
    capability_cols: list,
    lam: float = 1.0,
    seed: int = 42,
    pool: Literal["add", "geom", "softmin"] = "add",
    tau: float = 1.0,
    normalize: bool = False,
    mu_c: float = 2.5,
    sigma_c: float = 0.8,
    sigma_pop_sd: float = 0.5,
    fix_kappa: bool = True,
    **model_kwargs,
) -> Tuple[az.InferenceData, pm.Model]:
    """
    Fit the hierarchical population model for all participants in performance_df.

    Args:
        performance_df: DataFrame with participants as rows, items as columns (0/1/NaN)
        demand_matrix: Array of shape (J, K) aligned with item_index
        item_index: Item identifiers matching performance_df columns
        capability_cols: List of K capability names in model parameter order
        lam: Per-level log step
        seed: Random seed
        pool: Pooling method
        tau: Temperature for softmin/geom
        normalize: If True, use mean instead of sum for additive pooling
        mu_c: Prior mean for population capabilities
        sigma_c: Prior SD for population capability means
        sigma_pop_sd: Scale for HalfNormal on within-population spread
        fix_kappa: If True, fix kappa=1 (recommended)

    Returns:
        Tuple of (InferenceData, Model)
    """
    participant_names = list(performance_df.index)
    Y_matrix = performance_df[item_index].to_numpy(float)  # (N, J)

    model = build_population_model(
        Y_matrix=Y_matrix,
        D=demand_matrix,
        participant_names=participant_names,
        capability_cols=capability_cols,
        lam=lam,
        mu_c=mu_c,
        sigma_c=sigma_c,
        sigma_pop_sd=sigma_pop_sd,
        alpha_prior=model_kwargs.get("alpha_prior", (0.0, 0.5)),
        pool=pool,
        tau=tau,
        normalize=normalize,
        fix_kappa=fix_kappa,
    )

    idata = fit_model(model, seed=seed)
    return idata, model


def extract_capability_samples(
    idata: az.InferenceData,
    var: str = "c",
    draws: Optional[int] = 2000,
    chains: Optional[list] = None,
) -> np.ndarray:
    """
    Extract posterior samples for a variable from InferenceData.

    Args:
        idata: ArviZ InferenceData object
        var: Variable name to extract
        draws: Number of draws to sample (None for all)
        chains: List of chains to include (None for all)

    Returns:
        Array of shape (n_samples, n_capabilities)
    """
    x = idata.posterior[var]
    var_dim_name = [d for d in x.dims if d not in ["chain", "draw"]][0]

    if chains is not None:
        x = x.sel(chain=chains)

    x = x.stack(sample=("chain", "draw")).transpose("sample", var_dim_name).values

    if draws is not None and draws < x.shape[0]:
        rng = np.random.default_rng(0)
        idx = rng.choice(x.shape[0], size=draws, replace=False)
        x = x[idx]

    return x


def extract_population_capability_samples(
    idata: az.InferenceData,
    participant_names: list,
    draws: Optional[int] = 2000,
) -> dict:
    """
    Extract posterior capability samples for each participant from a population model.

    Returns a dict compatible with the existing suitability pipeline
    (same format as passing individual-model InferenceData to score_all_tasks).

    Args:
        idata: InferenceData from build_population_model
        participant_names: List of participant names (must match model coords)
        draws: Number of posterior draws to subsample (None for all)

    Returns:
        Dict mapping participant name -> InferenceData with a synthetic
        single-agent posterior containing that participant's c samples.
        Population-level mu_pop is also available as "population" key.
    """
    import xarray as xr

    c_samples = idata.posterior["c"]  # (chain, draw, participant, capability)
    c_stacked = c_samples.stack(sample=("chain", "draw"))
    # c_stacked has dims (participant, capability, sample)

    if draws is not None:
        n_total = c_stacked.sizes["sample"]
        if draws < n_total:
            rng = np.random.default_rng(0)
            idx = rng.choice(n_total, size=draws, replace=False)
            c_stacked = c_stacked.isel(sample=idx)

    result = {}

    # Per-participant: create a minimal InferenceData wrapping that participant's c
    for name in participant_names:
        c_p = c_stacked.sel(participant=name)  # (capability, sample)
        # Reshape back to (chain=1, draw=n_draws, capability)
        c_np = c_p.transpose("sample", "capability").values  # (n_draws, K)
        c_xr = xr.DataArray(
            c_np[None, :, :],  # (1, n_draws, K)
            dims=["chain", "draw", "capability_dim_0"],
            coords={"capability_dim_0": c_samples.coords["capability"].values},
        )
        synthetic_posterior = xr.Dataset({"c": c_xr})
        result[name] = az.InferenceData(posterior=synthetic_posterior)

    # Population mean as a synthetic agent
    mu_pop = idata.posterior["mu_pop"]  # (chain, draw, capability)
    mu_stacked = mu_pop.stack(sample=("chain", "draw"))
    if draws is not None and draws < mu_stacked.sizes["sample"]:
        mu_stacked = mu_stacked.isel(sample=idx)
    mu_np = mu_stacked.transpose("sample", "capability").values[None, :, :]
    mu_xr = xr.DataArray(
        mu_np,
        dims=["chain", "draw", "capability_dim_0"],
        coords={"capability_dim_0": c_samples.coords["capability"].values},
    )
    result["population_mean"] = az.InferenceData(
        posterior=xr.Dataset({"c": mu_xr})
    )

    return result


def collect_capability_means(
    agent_idata: dict,
    model_capability_cols: list,
    display_capability_cols: Optional[list] = None,
    use_theta: bool = False,
) -> pd.DataFrame:
    """
    Extract posterior mean capability levels for multiple agents.

    Args:
        agent_idata: Dictionary mapping agent names to InferenceData objects
        model_capability_cols: List of capability names in model parameter order
                               (i.e., c[0] corresponds to model_capability_cols[0])
        display_capability_cols: List of capability names in desired output order.
                                 If None, uses model_capability_cols order.
        use_theta: If True, return theta = exp(c), the ratio-scale capability.
                   Note: theta is the conceptually primary quantity; c = log(theta) is
                   the log-scale parameterization used for inference.

    Returns:
        DataFrame with agents as rows and capabilities as columns (in display order)
    """
    if display_capability_cols is None:
        display_capability_cols = model_capability_cols

    data = {}
    for agent, idata in agent_idata.items():
        if use_theta and "theta" in idata.posterior:
            means = idata.posterior["theta"].mean(dim=("chain", "draw")).to_numpy()
        else:
            means = idata.posterior["c"].mean(dim=("chain", "draw")).to_numpy()
            if use_theta:
                means = np.exp(means)

        # Create series with model order, then reindex to display order
        model_series = pd.Series(means, index=model_capability_cols)
        data[agent] = model_series.reindex(display_capability_cols)

    return pd.DataFrame.from_dict(data, orient="index")


def collect_capability_summaries(
    agent_idata: dict,
    model_capability_cols: list,
    display_capability_cols: Optional[list] = None,
    use_theta: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract posterior summaries (mean, lower CI, upper CI) for capability levels.

    Args:
        agent_idata: Dictionary mapping agent names to InferenceData objects
        model_capability_cols: List of capability names in model parameter order
        display_capability_cols: List of capability names in desired output order.
                                 If None, uses model_capability_cols order.
        use_theta: If True, return theta = exp(c), the ratio-scale capability.

    Returns:
        Tuple of (mean_df, ci_lower_df, ci_upper_df) in display order
    """
    if display_capability_cols is None:
        display_capability_cols = model_capability_cols

    mean_data = {}
    lo_data = {}
    hi_data = {}

    for agent, idata in agent_idata.items():
        if use_theta and "theta" in idata.posterior:
            samples = idata.posterior["theta"]
        else:
            samples = idata.posterior["c"]
            if use_theta:
                samples = np.exp(samples)

        means = samples.mean(dim=("chain", "draw")).to_numpy()
        lo = samples.quantile(0.025, dim=("chain", "draw")).to_numpy()
        hi = samples.quantile(0.975, dim=("chain", "draw")).to_numpy()

        # Create series with model order, then reindex to display order
        mean_data[agent] = pd.Series(means, index=model_capability_cols).reindex(display_capability_cols)
        lo_data[agent] = pd.Series(lo, index=model_capability_cols).reindex(display_capability_cols)
        hi_data[agent] = pd.Series(hi, index=model_capability_cols).reindex(display_capability_cols)

    mean_df = pd.DataFrame.from_dict(mean_data, orient="index")
    lo_df = pd.DataFrame.from_dict(lo_data, orient="index")
    hi_df = pd.DataFrame.from_dict(hi_data, orient="index")

    return mean_df, lo_df, hi_df
