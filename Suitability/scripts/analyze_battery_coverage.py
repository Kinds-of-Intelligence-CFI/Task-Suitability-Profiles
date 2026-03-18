#!/usr/bin/env python
"""
Analyse the annotated item battery to inform short-form human battery design.

Produces four plots:
  1. Coverage – items with high demand (D>=3) per capability, total vs primary
  2. Co-occurrence heatmap – capability demand correlation structure
  3. Demand distribution – violin of D values per capability
  4. Rank curve – how D-matrix rank grows with number of items (random vs greedy)

Usage:
    python scripts/analyze_battery_coverage.py \
        --annotations data/processed/annotations.csv \
        --abilities config/abilities.csv \
        --exclude ProsM OP EPaE \
        --output figures/battery_coverage

    # To also save a CSV of the coverage table:
    python scripts/analyze_battery_coverage.py ... --save-table
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

DEMAND_THRESHOLD = 3   # D >= this counts as "high demand"
RANK_RANDOM_REPS = 200
RANK_SAMPLE_SIZES = list(range(5, 501, 5))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_short_name_map(abilities_df: pd.DataFrame, ann_cols: list[str]) -> dict[str, str]:
    """Map annotation column names -> acronyms from abilities.csv."""
    name_to_acr = dict(zip(abilities_df["Abilities"], abilities_df["Acronym"]))
    result = {}
    for col in ann_cols:
        # Try exact match first
        if col in name_to_acr:
            result[col] = name_to_acr[col]
            continue
        # Try case-insensitive substring match
        col_lower = col.lower()
        matched = None
        for full, acr in name_to_acr.items():
            if col_lower in full.lower() or full.lower() in col_lower:
                matched = acr
                break
        result[col] = matched if matched else col[:6]
    return result


def coverage_stats(D: np.ndarray, cap_cols: list[str]) -> pd.DataFrame:
    """Per-capability coverage statistics."""
    rows = []
    row_max = D.max(axis=1)
    for i, cap in enumerate(cap_cols):
        d_k = D[:, i]
        high = d_k >= DEMAND_THRESHOLD
        primary = high & (d_k == row_max)
        other_sum = D.sum(axis=1) - d_k
        isolated = high & (other_sum <= d_k)
        rows.append({
            "capability": cap,
            "total_items": len(d_k),
            "high_demand": int(high.sum()),
            "primary": int(primary.sum()),
            "isolated": int(isolated.sum()),
            "mean_D": float(d_k.mean()),
            "pct_nonzero": float((d_k > 0).mean() * 100),
        })
    return pd.DataFrame(rows).set_index("capability")


def _min_sv(D_sub: np.ndarray, n_caps: int) -> float:
    """
    Minimum singular value of D_sub across all n_caps dimensions.

    Returns 0 when the matrix is rank-deficient (fewer items than capabilities),
    because the missing dimensions have zero singular value.
    """
    m = D_sub.shape[0]
    if m == 0 or m < n_caps:
        return 0.0
    sv = np.linalg.svd(D_sub, compute_uv=False, full_matrices=False)
    # sv has length min(m, n_caps); pad with zeros if needed
    if len(sv) < n_caps:
        return 0.0
    return float(sv[-1])


def rank_random_curve(
    D: np.ndarray,
    sample_sizes: list[int],
    n_reps: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mean ± std of matrix rank and min singular value over random subsets."""
    n_caps = D.shape[1]
    rank_means, rank_stds = [], []
    sv_means, sv_stds = [], []
    n_items = D.shape[0]
    for n in sample_sizes:
        ranks, svs = [], []
        for _ in range(n_reps):
            idx = rng.choice(n_items, size=min(n, n_items), replace=False)
            sub = D[idx]
            ranks.append(np.linalg.matrix_rank(sub))
            svs.append(_min_sv(sub, n_caps))
        rank_means.append(np.mean(ranks))
        rank_stds.append(np.std(ranks))
        sv_means.append(np.mean(svs))
        sv_stds.append(np.std(svs))
    return np.array(rank_means), np.array(rank_stds), np.array(sv_means), np.array(sv_stds)


def rank_greedy_curve(D: np.ndarray, max_items: int) -> tuple[list[int], list[float]]:
    """
    Greedy item selection maximising matrix rank.

    At each step, adds the item that maximally increases the trace of D^T D
    (proxy for information gain). Returns (ranks, min_singular_values).
    """
    n_items, n_caps = D.shape
    selected = []
    ranks, svs = [], []
    D_selected = np.empty((0, n_caps))

    item_info = (D ** 2).sum(axis=1)
    remaining = set(range(n_items))

    for _ in range(max_items):
        if not remaining:
            break
        best = max(remaining, key=lambda j: item_info[j])
        selected.append(best)
        remaining.discard(best)
        D_selected = np.vstack([D_selected, D[best]])
        ranks.append(np.linalg.matrix_rank(D_selected))
        svs.append(_min_sv(D_selected, n_caps))

    return ranks, svs


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_coverage(
    stats: pd.DataFrame,
    short_names: dict,
    save_path: Path,
    show: bool,
) -> None:
    caps = stats.index.tolist()
    short = [short_names.get(c, c) for c in caps]
    high = stats["high_demand"].values
    primary = stats["primary"].values

    order = np.argsort(primary)
    caps_ord = [caps[i] for i in order]
    short_ord = [short[i] for i in order]
    high_ord = high[order]
    primary_ord = primary[order]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(caps_ord))
    ax.barh(y, high_ord, color="#90CAF9", label=f"D ≥ {DEMAND_THRESHOLD}")
    ax.barh(y, primary_ord, color="#1565C0", label="Primary demand")
    ax.set_yticks(y)
    ax.set_yticklabels(short_ord, fontsize=9)
    ax.set_xlabel("Number of items")
    ax.set_title("Item coverage per capability")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_cooccurrence(
    D: np.ndarray,
    cap_cols: list[str],
    short_names: dict,
    save_path: Path,
    show: bool,
) -> None:
    short = [short_names.get(c, c) for c in cap_cols]
    corr = np.corrcoef(D.T)

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(len(short)))
    ax.set_yticks(range(len(short)))
    ax.set_xticklabels(short, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short, fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Capability demand correlation (co-occurrence structure)")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_demand_distribution(
    D: np.ndarray,
    cap_cols: list[str],
    short_names: dict,
    save_path: Path,
    show: bool,
) -> None:
    short = [short_names.get(c, c) for c in cap_cols]
    order = np.argsort([D[:, i].mean() for i in range(len(cap_cols))])

    fig, ax = plt.subplots(figsize=(10, 5))
    data_ord = [D[:, i] for i in order]
    short_ord = [short[i] for i in order]

    parts = ax.violinplot(data_ord, positions=range(len(cap_cols)), showmedians=True)
    for pc in parts["bodies"]:
        pc.set_facecolor("#90CAF9")
        pc.set_alpha(0.7)
    ax.axhline(DEMAND_THRESHOLD, color="red", linestyle="--", linewidth=0.8,
               label=f"D = {DEMAND_THRESHOLD} threshold")
    ax.set_xticks(range(len(cap_cols)))
    ax.set_xticklabels(short_ord, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Demand value (D)")
    ax.set_title("Distribution of demand values per capability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_rank_curve(
    D: np.ndarray,
    n_caps: int,
    sample_sizes: list[int],
    save_path: Path,
    show: bool,
    seed: int = 42,
) -> None:
    rng = np.random.default_rng(seed)
    print("  Computing random rank curves...")
    rand_rank_mean, rand_rank_std, rand_sv_mean, rand_sv_std = rank_random_curve(
        D, sample_sizes, RANK_RANDOM_REPS, rng
    )

    max_greedy = max(sample_sizes)
    print("  Computing greedy rank curve...")
    greedy_ranks, greedy_svs = rank_greedy_curve(D, max_greedy)
    greedy_x = list(range(1, len(greedy_ranks) + 1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    # Panel 1: rank
    ax1.fill_between(
        sample_sizes, rand_rank_mean - rand_rank_std, rand_rank_mean + rand_rank_std,
        alpha=0.25, color="#90CAF9", label="Random ± 1 SD",
    )
    ax1.plot(sample_sizes, rand_rank_mean, color="#1565C0", label="Random (mean)")
    ax1.plot(greedy_x, greedy_ranks, color="#E53935", label="Greedy (trace-max)")
    ax1.axhline(n_caps, color="black", linestyle="--", linewidth=0.8,
                label=f"Full rank ({n_caps})")
    ax1.set_ylabel("D-matrix rank")
    ax1.set_title("Identifiability: D-matrix rank vs. number of selected items")
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, n_caps + 1)

    # Panel 2: min singular value (normalised by sqrt(n_items))
    ax2.fill_between(
        sample_sizes, rand_sv_mean - rand_sv_std, rand_sv_mean + rand_sv_std,
        alpha=0.25, color="#90CAF9", label="Random ± 1 SD",
    )
    ax2.plot(sample_sizes, rand_sv_mean, color="#1565C0", label="Random (mean)")
    ax2.plot(greedy_x[:len(greedy_svs)], greedy_svs, color="#E53935",
             label="Greedy (trace-max)")
    ax2.set_xlabel("Number of items selected")
    ax2.set_ylabel("Min singular value of D")
    ax2.set_title("Precision: minimum singular value (higher = better capability separation)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_difficulty_spread(
    D: np.ndarray,
    cap_cols: list[str],
    short_names: dict,
    save_path: Path,
    show: bool,
    target_n: int = 150,
) -> None:
    """
    For each capability, show the count of primary items at each demand level (1-5).

    "Primary" = this capability has the highest demand in the item.
    This reveals whether stratified selection can find items across the full
    difficulty range for each capability.
    """
    n_caps = len(cap_cols)
    short = [short_names.get(c, c) for c in cap_cols]
    row_max = D.max(axis=1)
    items_per_cap = target_n // n_caps

    d_levels = [1, 2, 3, 4, 5]
    counts = np.zeros((n_caps, len(d_levels)), dtype=int)

    for i, cap in enumerate(cap_cols):
        d_k = D[:, i]
        is_primary = d_k == row_max
        for j, lev in enumerate(d_levels):
            counts[i, j] = int(((d_k == lev) & is_primary).sum())

    # Order capabilities by total primary items
    order = np.argsort(counts.sum(axis=1))
    counts_ord = counts[order]
    short_ord = [short[i] for i in order]

    colors = ["#BBDEFB", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0"]
    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(n_caps)
    left = np.zeros(n_caps)
    for j, (lev, col) in enumerate(zip(d_levels, colors)):
        ax.barh(y, counts_ord[:, j], left=left, color=col, label=f"D = {lev}")
        left += counts_ord[:, j]

    ax.axvline(items_per_cap, color="#E53935", linestyle="--", linewidth=1.2,
               label=f"Target per capability ({items_per_cap})")
    ax.set_yticks(y)
    ax.set_yticklabels(short_ord, fontsize=9)
    ax.set_xlabel("Number of primary items")
    ax.set_title(
        f"Difficulty spread within each capability's primary item pool\n"
        f"(target battery n={target_n}, ~{items_per_cap} items per capability)"
    )
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_nonzero_histogram(
    D: np.ndarray,
    save_path: Path,
    show: bool,
) -> None:
    counts = (D > 0).sum(axis=1)
    n_caps = D.shape[1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=range(0, n_caps + 2), align="left", color="#90CAF9",
            edgecolor="white", linewidth=0.5)
    ax.axvline(counts.mean(), color="#E53935", linestyle="--",
               label=f"Mean = {counts.mean():.1f}")
    ax.set_xlabel("Number of capabilities with D > 0")
    ax.set_ylabel("Number of items")
    ax.set_title("Item dimensionality: how many capabilities does each item demand?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse item battery coverage")
    parser.add_argument(
        "--annotations",
        default="data/processed/annotations.csv",
        help="Path to annotations CSV",
    )
    parser.add_argument(
        "--abilities",
        default="config/abilities.csv",
        help="Path to abilities CSV (for acronym mapping)",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=["ProsM", "OP", "EPaE"],
        help="Capability acronyms to exclude (default: ProsM OP EPaE)",
    )
    parser.add_argument(
        "--output",
        default="figures/battery_coverage",
        help="Output path prefix (directory/stem)",
    )
    parser.add_argument(
        "--save-table",
        action="store_true",
        help="Save coverage statistics as CSV",
    )
    parser.add_argument(
        "--target-n",
        type=int,
        default=150,
        help="Target battery size for difficulty spread plot (default: 150)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively",
    )
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    show = not args.no_show

    # Load data
    print("Loading annotations...")
    df = pd.read_csv(args.annotations)
    ann_cap_cols = df.columns[2:].tolist()

    abilities_df = pd.read_csv(args.abilities)
    short_names = build_short_name_map(abilities_df, ann_cap_cols)

    # Build exclusion set using full names
    acr_to_full = {v: k for k, v in short_names.items()}
    exclude_full = set()
    for acr in (args.exclude or []):
        if acr in acr_to_full:
            exclude_full.add(acr_to_full[acr])
        else:
            # Try matching directly against full names
            exclude_full.add(acr)

    cap_cols = [c for c in ann_cap_cols if c not in exclude_full]
    excluded = [c for c in ann_cap_cols if c in exclude_full]
    print(f"  Capabilities: {len(cap_cols)} retained, {len(excluded)} excluded")
    if excluded:
        excl_short = [short_names.get(c, c) for c in excluded]
        print(f"  Excluded: {', '.join(excl_short)}")

    D = df[cap_cols].to_numpy(float)
    n_caps = len(cap_cols)

    # Summary stats
    print(f"\nBattery summary ({n_caps} capabilities, {len(D)} items):")
    print(f"  Mean non-zero demands per item: {(D > 0).sum(axis=1).mean():.2f}")
    print(f"  Mean D >= {DEMAND_THRESHOLD} per item:   {(D >= DEMAND_THRESHOLD).sum(axis=1).mean():.2f}")

    stats = coverage_stats(D, cap_cols)
    print(f"\n{'Capability':<45} {'D>=3':>6} {'Primary':>8}")
    print("-" * 62)
    for cap, row in stats.sort_values("primary").iterrows():
        print(f"  {short_names.get(cap, cap):<43} {row['high_demand']:>6} {row['primary']:>8}")

    if args.save_table:
        table_path = out.parent / f"{out.stem}_table.csv"
        stats["short_name"] = [short_names.get(c, c) for c in stats.index]
        stats.to_csv(table_path)
        print(f"\nCoverage table saved to {table_path}")

    # Plot 1: Coverage
    print("\nPlot 1: Coverage...")
    plot_coverage(stats, short_names, out.parent / f"{out.stem}_coverage.png", show)

    # Plot 2: Co-occurrence heatmap
    print("Plot 2: Co-occurrence heatmap...")
    plot_cooccurrence(D, cap_cols, short_names, out.parent / f"{out.stem}_cooccurrence.png", show)

    # Plot 3: Demand distributions
    print("Plot 3: Demand distributions...")
    plot_demand_distribution(D, cap_cols, short_names, out.parent / f"{out.stem}_demand_dist.png", show)

    # Plot 4: Non-zero histogram
    print("Plot 4: Item dimensionality histogram...")
    plot_nonzero_histogram(D, out.parent / f"{out.stem}_dimensionality.png", show)

    # Plot 5: Difficulty spread
    print("Plot 5: Difficulty spread...")
    plot_difficulty_spread(
        D, cap_cols, short_names,
        out.parent / f"{out.stem}_difficulty_spread.png",
        show,
        target_n=args.target_n,
    )

    # Plot 6: Rank curve
    print("Plot 6: Rank curve...")
    plot_rank_curve(D, n_caps, RANK_SAMPLE_SIZES, out.parent / f"{out.stem}_rank_curve.png", show)

    print("\nDone.")


if __name__ == "__main__":
    main()
