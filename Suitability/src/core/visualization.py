"""
Visualization functions for capability profiles and model diagnostics.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
from scipy.special import expit as sigmoid
from typing import Optional, List, Tuple


def plot_radar_capabilities(
    cap_df: pd.DataFrame,
    capability_info: pd.DataFrame,
    overlay: bool = True,
    agents: Optional[List[str]] = None,
    rlim: Optional[Tuple[float, float]] = None,
    fill_alpha: float = 0.15,
    figsize: Tuple[int, int] = (8, 8),
    title: str = "Capability Profiles (θ scale)",
    save_path: Optional[str] = None,
    dpi: int = 300,
    label_map: Optional[dict] = None,
    use_theta: bool = True,
    clip_zero: bool = True,
    show: bool = True,
) -> None:
    """
    Plot radar (spider) charts of capability means per agent.

    By default, uses theta scale (exp(c)) which is always positive and suitable
    for radar plots. The theta scale represents capability on a ratio scale where
    theta=1 is a neutral reference point.

    Args:
        cap_df: DataFrame with agents as rows and capabilities as columns (c values)
        capability_info: DataFrame with 'Abilities' and 'Acronym' columns for labeling
        overlay: If True, plot all agents on one chart; if False, separate subplots
        agents: List of agent names to plot (default: all)
        rlim: Tuple of (min, max) for radial axis. If None, auto-computed.
        fill_alpha: Alpha for filled area
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        dpi: DPI for raster output
        label_map: Dictionary to remap agent names in legend
        use_theta: If True (default), exponentiate values (c -> exp(c)) for ratio scale.
                   Recommended for radar plots as theta is always positive.
        clip_zero: If True (default), clip values below 0 to 0 before plotting.
                   Negative c values are meaningful (below minimum competence) but
                   cannot be shown on a radar axis; clipping preserves the shape
                   for positive capabilities while collapsing negatives to the origin.
        show: If True, display the plot interactively
    """
    # Capability order and short-name mapping
    capability_order = capability_info["Abilities"].tolist()
    short_labels = dict(zip(capability_order, capability_info["Acronym"]))

    # Filter to capabilities present in data and with non-NaN values
    # (capabilities not in the inferred subset will be all-NaN after reindex)
    available_caps = [c for c in capability_order if c in cap_df.columns and cap_df[c].notna().any()]
    labels = [short_labels[c] for c in available_caps]
    cap_df = cap_df[available_caps].copy()

    if agents is None:
        agents = list(cap_df.index)

    # Transform to theta scale (default for radar plots)
    if use_theta:
        cap_df = np.exp(cap_df)
        if title == "Capability Profiles (θ scale)":
            pass  # Keep default title
    else:
        if title == "Capability Profiles (θ scale)":
            title = "Capability Profiles (log scale)"

    # Clip negatives to 0 (negatives are meaningful but cannot be shown on radar axes)
    if clip_zero:
        cap_df = cap_df.clip(lower=0)

    # Auto-compute rlim if not provided
    if rlim is None:
        data_min = cap_df.values.min()
        data_max = cap_df.values.max()
        # Add some padding
        rlim = (max(0, data_min * 0.8), data_max * 1.2)

    K = len(labels)
    angles = np.linspace(0, 2 * np.pi, K, endpoint=False)
    angles = np.concatenate([angles, angles[:1]])  # close polygon

    def close(vals):
        vals = np.asarray(vals, float)
        return np.concatenate([vals, vals[:1]])

    if overlay:
        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)
        for agent in agents:
            vals = cap_df.loc[agent, :].values
            ax.plot(angles, close(vals), linewidth=2, label=str(agent))
            ax.fill(angles, close(vals), alpha=fill_alpha)
        ax.set_ylim(*rlim)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=14)
        yticks = np.arange(np.floor(rlim[0]), np.ceil(rlim[1]) + 1, 1)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{int(t)}" for t in yticks], fontsize=11)
        ax.set_title(title, pad=40, fontsize=16, weight="bold")

        # Legend with optional label mapping
        handles, legend_labels = ax.get_legend_handles_labels()
        if label_map:
            legend_labels = [label_map.get(lbl, lbl) for lbl in legend_labels]
        ax.legend(
            handles, legend_labels,
            loc="upper right",
            bbox_to_anchor=(1.3, 1.15),
            fontsize=14,
            frameon=False,
        )
        plt.tight_layout()

    else:
        # Separate subplot per agent
        n = len(agents)
        ncols = 3 if n >= 3 else n
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(
            nrows, ncols,
            subplot_kw=dict(polar=True),
            figsize=(ncols * 5, nrows * 5),
        )
        axes = np.atleast_1d(axes).ravel()

        for ax, agent in zip(axes, agents):
            vals = cap_df.loc[agent, :].values
            ax.plot(angles, close(vals), linewidth=2, label=str(agent))
            ax.fill(angles, close(vals), alpha=fill_alpha)
            ax.set_ylim(*rlim)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, fontsize=14)
            yticks = np.arange(np.floor(rlim[0]), np.ceil(rlim[1]) + 1, 1)
            ax.set_yticks(yticks)
            ax.set_yticklabels([f"{int(t)}" for t in yticks], fontsize=11)
            display_name = label_map.get(agent, agent) if label_map else agent
            ax.set_title(display_name, pad=20, fontsize=14, weight="bold")

        for ax in axes[len(agents):]:
            ax.set_visible(False)

        plt.suptitle(title, y=0.98, fontsize=16, weight="bold")
        plt.tight_layout()

    # Save
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".pdf", ".svg"]:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved radar plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_demand_distribution(
    D: np.ndarray,
    capability_cols: List[str],
    abilities_df: Optional[pd.DataFrame] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Demand Level Distribution by Capability",
    save_path: Optional[str] = None,
    dpi: int = 300,
    show_counts: bool = True,
    max_level: int = 5,
    show: bool = True,
) -> pd.DataFrame:
    """
    Plot the distribution of demand levels across capabilities.

    Shows a stacked bar chart with the count/proportion of each demand level
    (0, 1, 2, 3, 4, 5) for each capability.

    Args:
        D: Demand matrix of shape (J, K) where J is items, K is capabilities
        capability_cols: List of capability column names
        abilities_df: Optional DataFrame with 'Abilities' and 'Acronym' columns for labels
        figsize: Figure size
        title: Plot title
        save_path: Path to save figure (optional)
        dpi: DPI for raster output
        show_counts: If True, show counts; if False, show proportions
        max_level: Maximum demand level to consider (default 5)

    Returns:
        DataFrame with demand level counts per capability
    """
    J, K = D.shape

    # Get labels
    if abilities_df is not None:
        acronym_map = dict(zip(abilities_df["Abilities"], abilities_df["Acronym"]))
        labels = [acronym_map.get(c, c) for c in capability_cols]
    else:
        labels = capability_cols

    # Count demand levels for each capability
    levels = list(range(max_level + 1))
    counts = np.zeros((len(levels), K), dtype=int)

    for level in levels:
        counts[level, :] = (D == level).sum(axis=0)

    # Create DataFrame for return
    counts_df = pd.DataFrame(
        counts.T,
        index=labels,
        columns=[f"Level {l}" for l in levels],
    )

    # Compute proportions
    props = counts / J

    # Sort by coverage (proportion of non-zero demands)
    coverage = 1 - props[0, :]  # 1 - proportion of zeros
    sort_idx = np.argsort(coverage)[::-1]  # Descending

    sorted_labels = [labels[i] for i in sort_idx]
    sorted_counts = counts[:, sort_idx]
    sorted_props = props[:, sort_idx]

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(K)
    width = 0.8

    # Color map: Level 0 is grey, then increasingly saturated colors
    colors = ["#d0d0d0"]  # Grey for level 0
    cmap = plt.cm.Blues
    for i in range(1, max_level + 1):
        colors.append(cmap(0.3 + 0.7 * i / max_level))

    # Stacked bar chart
    data = sorted_props if not show_counts else sorted_counts / J
    bottom = np.zeros(K)

    for level in levels:
        ax.bar(
            x,
            data[level, :],
            width,
            bottom=bottom,
            label=f"Level {level}",
            color=colors[level],
            edgecolor="white",
            linewidth=0.5,
        )
        bottom += data[level, :]

    ax.set_xticks(x)
    ax.set_xticklabels(sorted_labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Proportion of Items")
    ax.set_xlabel("Capability")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylim(0, 1)
    ax.legend(
        title="Demand Level",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )

    # Add coverage annotation
    for i, idx in enumerate(sort_idx):
        cov = coverage[idx]
        ax.annotate(
            f"{cov*100:.0f}%",
            xy=(i, 1.01),
            ha="center",
            va="bottom",
            fontsize=8,
            color="darkblue",
        )

    ax.text(
        0.5, 1.08,
        "Coverage (% non-zero)",
        transform=ax.transAxes,
        ha="center",
        fontsize=9,
        color="darkblue",
    )

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".pdf", ".svg"]:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved demand distribution plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()

    return counts_df


def plot_icc_curve(
    idata: az.InferenceData,
    capability_cols: List[str],
    D: np.ndarray,
    k: int,
    lam: float = 1.0,
    include_other: bool = False,
    figsize: Tuple[int, int] = (5, 3),
    save_path: Optional[str] = None,
    show: bool = False,
) -> None:
    """
    Plot an Item Characteristic Curve (ICC) for a specific capability.

    Shows the probability of success as a function of demand level for capability k.

    Args:
        idata: ArviZ InferenceData with posterior samples
        capability_cols: List of capability column names
        D: Demand matrix of shape (J, K)
        k: Index of the capability to plot
        lam: Per-level log step
        include_other: If True, include average contribution from other capabilities
        figsize: Figure size
        save_path: Path to save figure (optional)
        show: If True, display the plot interactively
    """
    # Posterior means
    c_mean = idata.posterior["c"].mean(dim=("chain", "draw")).to_numpy()
    if "kappa" in idata.posterior:
        kap_mean = idata.posterior["kappa"].mean(dim=("chain", "draw")).to_numpy()
    else:
        kap_mean = np.ones(len(c_mean))
    alpha = float(idata.posterior["alpha"].mean(dim=("chain", "draw")))

    K = len(c_mean)
    lam_arr = np.broadcast_to(lam, K)

    c_k, kap_k, lam_k = c_mean[k], kap_mean[k], lam_arr[k]

    # Mean demand per dimension
    d_mean = np.mean(D, axis=0)

    # Optional average contribution from other capabilities
    if include_other:
        avg_other = np.mean([
            kap_mean[i] * (c_mean[i] - lam_arr[i] * d_mean[i])
            for i in range(K) if i != k
        ])
    else:
        avg_other = 0.0

    # Compute curve
    d_grid = np.linspace(0, 5, 61)
    is_on = (d_grid > 0).astype(float)
    margin = is_on * (c_k - lam_k * d_grid)
    z = alpha + avg_other + kap_k * margin
    p = sigmoid(z)

    # Plot
    plt.figure(figsize=figsize)
    plt.plot(d_grid, p, lw=2)
    plt.xlabel("Demand level d")
    plt.ylabel("P(success)")
    plt.title(
        f"ICC for {capability_cols[k]}"
        + (" (with avg other dims)" if include_other else "")
    )
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved ICC plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_suitability_scores(
    mean_df: pd.DataFrame,
    ci_lo_df: pd.DataFrame,
    ci_hi_df: pd.DataFrame,
    figsize: Tuple[int, int] = (14, 6),
    title: str = "Task scores by agent",
    sort_tasks: bool = False,
    jitter: float = 0.1,
    seed: int = 42,
    save_path: Optional[str] = None,
    dpi: int = 300,
    label_map: Optional[dict] = None,
    ylim: Optional[Tuple[float, float]] = (0, 5),
    show: bool = True,
) -> None:
    """
    Plot suitability scores for all tasks with error bars.

    Args:
        mean_df: DataFrame of mean scores (tasks x agents)
        ci_lo_df: DataFrame of lower CI bounds
        ci_hi_df: DataFrame of upper CI bounds
        figsize: Figure size
        title: Plot title
        sort_tasks: If True, sort tasks by mean score
        jitter: Amount of horizontal jitter for points
        seed: Random seed for jitter
        save_path: Path to save figure (optional)
        dpi: DPI for raster output
        label_map: Dictionary to remap agent names in legend
        ylim: Y-axis limits as (min, max). Default (0, 5) for comparability across plots.
              Set to None for auto-scaling.
        show: If True, display the plot interactively
    """
    rng = np.random.default_rng(seed)
    tasks = list(mean_df.index)
    agents = list(mean_df.columns)

    if sort_tasks:
        order = mean_df.mean(axis=1).sort_values(ascending=False).index
        mean_df = mean_df.loc[order]
        ci_lo_df = ci_lo_df.loc[order]
        ci_hi_df = ci_hi_df.loc[order]
        tasks = list(order)

    x = np.arange(len(tasks))
    plt.figure(figsize=figsize)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, agent in enumerate(agents):
        means = mean_df[agent].values
        los = ci_lo_df[agent].values
        his = ci_hi_df[agent].values
        err_lo = means - los
        err_hi = his - means
        xjit = x + rng.uniform(-jitter, jitter, size=len(x))

        plt.errorbar(
            xjit, means, yerr=[err_lo, err_hi],
            fmt="o-", capsize=3, label=agent,
            color=colors[i % len(colors)], alpha=0.9,
            markersize=5, lw=2,
        )

    plt.xticks(x, tasks, rotation=45, ha="right")
    plt.ylabel("Suitability Score (capability level)")
    plt.title(title, fontsize=14, weight="bold")
    plt.grid(alpha=0.3, axis="y")
    if ylim is not None:
        plt.ylim(*ylim)

    # Legend
    handles, labels = plt.gca().get_legend_handles_labels()
    if label_map:
        labels = [label_map.get(lbl, lbl) for lbl in labels]
    plt.legend(
        handles, labels,
        title="Agent",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
    )
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        ext = os.path.splitext(save_path)[1].lower()
        if ext in [".pdf", ".svg"]:
            plt.savefig(save_path, bbox_inches="tight")
        else:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved figure: {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def plot_forest(
    idata: az.InferenceData,
    var_names: List[str] = ["c", "kappa"],
    combined: bool = True,
    save_path: Optional[str] = None,
    abilities_df: Optional[pd.DataFrame] = None,
    figsize: Optional[Tuple[int, int]] = None,
    coverage: Optional[np.ndarray] = None,
    capability_cols: Optional[List[str]] = None,
    low_coverage_threshold: float = 0.05,
    show: bool = True,
) -> None:
    """
    Plot a forest plot of posterior distributions.

    Args:
        idata: ArviZ InferenceData
        var_names: Variables to include
        combined: If True, combine chains
        save_path: Path to save figure (optional)
        abilities_df: DataFrame with 'Abilities' and 'Acronym' columns for labeling.
                      If provided, replaces c[0], c[1], ... with acronyms.
        figsize: Figure size (optional)
        coverage: Array of coverage proportions per capability (0-1). If provided,
                  coverage is shown in labels and low-coverage capabilities are flagged.
        capability_cols: List of capability column names in the same order as coverage array.
                         Required if coverage is provided to ensure correct alignment.
        low_coverage_threshold: Capabilities with coverage below this are flagged (default 0.05)
    """
    # Build labels and reorder to abilities_df order for display
    if abilities_df is not None:
        # Get mapping from ability name to acronym
        name_to_acronym = dict(zip(abilities_df["Abilities"], abilities_df["Acronym"]))

        # Source order (from model/annotations) - if not provided, assume same as target
        full_target = abilities_df["Abilities"].tolist()
        source_order = capability_cols if capability_cols is not None else full_target

        # Target order for display: only include capabilities present in the model
        source_set = set(source_order)
        target_order = [c for c in full_target if c in source_set]

        # Build coverage lookup by capability name (source order)
        coverage_by_name = {}
        if coverage is not None:
            for i, cap_name in enumerate(source_order):
                coverage_by_name[cap_name] = coverage[i] if i < len(coverage) else 0

        # Create labels in target order (abilities_df order)
        labels = []
        for ability_name in target_order:
            acr = name_to_acronym.get(ability_name, ability_name[:4])
            if coverage_by_name:
                cov = coverage_by_name.get(ability_name, 0)
                cov_pct = cov * 100
                if cov < low_coverage_threshold:
                    labels.append(f"{acr} ({cov_pct:.0f}%) ⚠")
                else:
                    labels.append(f"{acr} ({cov_pct:.0f}%)")
            else:
                labels.append(acr)

        # Reorder idata from source order to target order, and apply labels
        idata = _reorder_and_relabel_idata(idata, source_order, target_order, labels)

    axes = az.plot_forest(idata, var_names=var_names, combined=combined, figsize=figsize)

    # If coverage provided, grey out low-coverage entries
    if coverage is not None and abilities_df is not None:
        ax = axes[0] if isinstance(axes, np.ndarray) else axes
        # Get y-tick labels and grey out low-coverage ones
        for i, label in enumerate(ax.get_yticklabels()):
            label_text = label.get_text()
            if "⚠" in label_text:
                label.set_color("gray")
                label.set_style("italic")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved forest plot to {save_path}")

    if show:
        plt.show()
    else:
        plt.close()


def _reorder_and_relabel_idata(
    idata: az.InferenceData,
    source_order: List[str],
    target_order: List[str],
    labels: List[str],
) -> az.InferenceData:
    """
    Reorder capability dimensions to match target order and apply labels.

    Args:
        idata: ArviZ InferenceData object
        source_order: List of capability names in the order they appear in idata (model order)
        target_order: List of capability names in the desired display order
        labels: List of labels to apply (in target_order)

    Returns:
        New InferenceData with reordered and relabeled coordinates
    """
    idata = idata.copy()

    # Build reorder index: for each position in target_order, find its position in source_order
    reorder_idx = [source_order.index(cap) for cap in target_order]

    # Find the capability dimension name (usually 'c_dim_0' or similar)
    if "c" in idata.posterior:
        c_dims = [d for d in idata.posterior["c"].dims if d not in ["chain", "draw"]]
        if c_dims:
            dim_name = c_dims[0]

            # Reorder and relabel the posterior
            idata.posterior = idata.posterior.isel({dim_name: reorder_idx})
            idata.posterior = idata.posterior.assign_coords({dim_name: labels})

    return idata
