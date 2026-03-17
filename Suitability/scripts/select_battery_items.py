#!/usr/bin/env python
"""
Select a stratified subset of items from the annotated battery for human testing.

Strategy:
  1. For each capability, identify items where that capability is the primary demand
     (highest D value in the row) at each demand level (1-5).
  2. Sample `--items-per-level` items per capability × D-level cell to ensure
     both capability coverage and difficulty spread.
  3. Fill any remaining budget (if cells are sparse) using E-optimal greedy
     selection to maximise the minimum singular value of the selected D sub-matrix.

Outputs:
  - CSV of selected items
  - Plots: difficulty spread, coverage, and min-SV comparison vs. random

Usage:
    python scripts/select_battery_items.py \
        --annotations data/processed/annotations.csv \
        --abilities config/abilities.csv \
        --exclude ProsM OP EPaE \
        --target-n 150 \
        --output data/processed/selected_battery
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DEMAND_LEVELS = [1, 2, 3, 4, 5]
SV_RANDOM_REPS = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_short_name_map(abilities_df: pd.DataFrame, ann_cols: list) -> dict:
    name_to_acr = dict(zip(abilities_df["Abilities"], abilities_df["Acronym"]))
    result = {}
    for col in ann_cols:
        if col in name_to_acr:
            result[col] = name_to_acr[col]
            continue
        col_lower = col.lower()
        matched = None
        for full, acr in name_to_acr.items():
            if col_lower in full.lower() or full.lower() in col_lower:
                matched = acr
                break
        result[col] = matched if matched else col[:6]
    return result


def _min_sv(D_sub: np.ndarray, n_caps: int) -> float:
    m = D_sub.shape[0]
    if m == 0 or m < n_caps:
        return 0.0
    sv = np.linalg.svd(D_sub, compute_uv=False, full_matrices=False)
    return float(sv[-1]) if len(sv) >= n_caps else 0.0


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def stratified_select(
    D: np.ndarray,
    cap_cols: list,
    items_per_level: int = 2,
    min_d: int = 1,
    rng: np.random.Generator = None,
) -> tuple[list[int], pd.DataFrame]:
    """
    For each capability × demand-level cell, sample up to `items_per_level` items
    where that capability is the primary demand (highest D in the row).

    Args:
        D: items × capabilities demand matrix
        cap_cols: capability column names
        items_per_level: target items per capability × D-level cell
        min_d: minimum D value for an item to be eligible
        rng: random number generator

    Returns:
        (selected_indices, allocation_df) where allocation_df records how many
        items were assigned per capability × D-level cell.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_caps = len(cap_cols)
    row_max = D.max(axis=1)
    selected = []
    selected_set = set()

    alloc_rows = []

    for i, cap in enumerate(cap_cols):
        d_k = D[:, i]
        is_primary = (d_k == row_max) & (d_k >= min_d)
        primary_idx = np.where(is_primary)[0]

        for lev in DEMAND_LEVELS:
            pool = [j for j in primary_idx if d_k[j] == lev and j not in selected_set]
            n_pick = min(items_per_level, len(pool))
            if n_pick > 0:
                picked = rng.choice(pool, size=n_pick, replace=False).tolist()
                selected.extend(picked)
                selected_set.update(picked)
            alloc_rows.append({
                "capability": cap,
                "d_level": lev,
                "pool_size": len(pool),
                "selected": n_pick,
            })

    alloc_df = pd.DataFrame(alloc_rows)
    return selected, alloc_df


def eoptimal_fill(
    D: np.ndarray,
    selected: list[int],
    target_n: int,
    n_caps: int,
    rng: np.random.Generator,
) -> list[int]:
    """
    Fill remaining budget to reach target_n by greedily selecting items that
    maximise the minimum singular value (E-optimal criterion).

    At each step: find current minimum singular vector, then add the item from
    the remaining pool that most strongly projects onto it.
    """
    n_fill = target_n - len(selected)
    if n_fill <= 0:
        return selected

    remaining = list(set(range(D.shape[0])) - set(selected))
    current = list(selected)

    for _ in range(n_fill):
        if not remaining:
            break

        D_cur = D[current]
        m = D_cur.shape[0]

        if m < n_caps:
            # Not yet full rank: pick the item maximising trace contribution
            info = (D[remaining] ** 2).sum(axis=1)
            best_pos = int(np.argmax(info))
        else:
            # Full rank: pick item that most increases min singular value
            _, _, Vt = np.linalg.svd(D_cur, full_matrices=False)
            v_min = Vt[-1]  # right singular vector for smallest sv
            projections = np.abs(D[remaining] @ v_min)
            best_pos = int(np.argmax(projections))

        best_idx = remaining.pop(best_pos)
        current.append(best_idx)

    return current


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def selection_report(
    D_sel: np.ndarray,
    D_full: np.ndarray,
    cap_cols: list,
    short_names: dict,
    alloc_df: pd.DataFrame,
    target_n: int,
) -> None:
    n_caps = len(cap_cols)
    print(f"\nSelected {len(D_sel)} / {target_n} items ({n_caps} capabilities)")
    print(f"  D-matrix rank:      {np.linalg.matrix_rank(D_sel)} / {n_caps}")
    print(f"  Min singular value: {_min_sv(D_sel, n_caps):.3f}")
    print(f"  Mean non-zero demands per item: {(D_sel > 0).sum(axis=1).mean():.2f}")

    print(f"\n{'':>6} {'':5}", end="")
    for lev in DEMAND_LEVELS:
        print(f"  D={lev}", end="")
    print(f"  {'total':>6}")
    print("-" * 55)

    for cap in cap_cols:
        short = short_names.get(cap, cap)
        row = alloc_df[alloc_df["capability"] == cap]
        print(f"  {short:<6}", end="")
        total = 0
        for lev in DEMAND_LEVELS:
            n = int(row[row["d_level"] == lev]["selected"].sum())
            total += n
            print(f"  {n:>4}", end="")
        print(f"  {total:>6}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_difficulty_spread_selected(
    D_sel: np.ndarray,
    cap_cols: list,
    short_names: dict,
    target_n: int,
    save_path: Path,
    show: bool,
) -> None:
    """Difficulty spread for selected items - mirrors analyze_battery_coverage plot."""
    n_caps = len(cap_cols)
    short = [short_names.get(c, c) for c in cap_cols]
    row_max = D_sel.max(axis=1)
    items_per_cap = target_n // n_caps

    counts = np.zeros((n_caps, len(DEMAND_LEVELS)), dtype=int)
    for i, cap in enumerate(cap_cols):
        d_k = D_sel[:, i]
        is_primary = d_k == row_max
        for j, lev in enumerate(DEMAND_LEVELS):
            counts[i, j] = int(((d_k == lev) & is_primary).sum())

    order = np.argsort(counts.sum(axis=1))
    counts_ord = counts[order]
    short_ord = [short[i] for i in order]

    colors = ["#BBDEFB", "#90CAF9", "#42A5F5", "#1E88E5", "#1565C0"]
    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(n_caps)
    left = np.zeros(n_caps)
    for j, (lev, col) in enumerate(zip(DEMAND_LEVELS, colors)):
        ax.barh(y, counts_ord[:, j], left=left, color=col, label=f"D = {lev}")
        left += counts_ord[:, j]

    ax.axvline(items_per_cap, color="#E53935", linestyle="--", linewidth=1.2,
               label=f"Target per capability ({items_per_cap})")
    ax.set_yticks(y)
    ax.set_yticklabels(short_ord, fontsize=9)
    ax.set_xlabel("Number of primary items")
    ax.set_title(f"Selected battery (n={len(D_sel)}): difficulty spread per capability")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_sv_comparison(
    D_sel: np.ndarray,
    D_full: np.ndarray,
    n_caps: int,
    save_path: Path,
    show: bool,
    seed: int = 42,
) -> None:
    """Compare min singular value of selected set vs. random samples of same size."""
    n_sel = len(D_sel)
    n_items = D_full.shape[0]
    rng = np.random.default_rng(seed)

    rand_svs = []
    for _ in range(SV_RANDOM_REPS):
        idx = rng.choice(n_items, size=n_sel, replace=False)
        rand_svs.append(_min_sv(D_full[idx], n_caps))
    rand_svs = np.array(rand_svs)

    selected_sv = _min_sv(D_sel, n_caps)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(rand_svs, bins=30, color="#90CAF9", edgecolor="white",
            label=f"Random n={n_sel} ({SV_RANDOM_REPS} reps)")
    ax.axvline(selected_sv, color="#E53935", linewidth=2,
               label=f"Selected (min SV = {selected_sv:.2f})")
    ax.axvline(rand_svs.mean(), color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Random mean ({rand_svs.mean():.2f})")
    percentile = float((rand_svs < selected_sv).mean() * 100)
    ax.set_xlabel("Min singular value of D")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Identifiability: selected set vs. random (n={n_sel})\n"
        f"Selected is at the {percentile:.0f}th percentile of random samples"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Select a stratified human battery subset from the annotated item pool"
    )
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
        "--target-n",
        type=int,
        default=150,
        help="Target battery size (default: 150)",
    )
    parser.add_argument(
        "--items-per-level",
        type=int,
        default=2,
        help="Items to sample per capability × D-level cell (default: 2)",
    )
    parser.add_argument(
        "--min-d",
        type=int,
        default=1,
        help="Minimum D value for primary item eligibility (default: 1)",
    )
    parser.add_argument(
        "--no-fill",
        action="store_true",
        help="Skip E-optimal fill step (return stratified selection only)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output",
        default="data/processed/selected_battery",
        help="Output path prefix (directory/stem) for CSV and plots",
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
    rng = np.random.default_rng(args.seed)

    # Load data
    print("Loading annotations...")
    df = pd.read_csv(args.annotations)
    ann_cap_cols = df.columns[2:].tolist()

    abilities_df = pd.read_csv(args.abilities)
    short_names = build_short_name_map(abilities_df, ann_cap_cols)

    # Resolve exclusions
    acr_to_full = {v: k for k, v in short_names.items()}
    exclude_full = set()
    for acr in (args.exclude or []):
        if acr in acr_to_full:
            exclude_full.add(acr_to_full[acr])
        else:
            exclude_full.add(acr)

    cap_cols = [c for c in ann_cap_cols if c not in exclude_full]
    print(f"  {len(cap_cols)} capabilities retained, {len(exclude_full)} excluded")

    D_full = df[cap_cols].to_numpy(float)
    n_caps = len(cap_cols)

    # Phase 1: stratified selection
    print(f"\nPhase 1: Stratified selection ({args.items_per_level} items per capability × D level)...")
    selected, alloc_df = stratified_select(
        D_full, cap_cols,
        items_per_level=args.items_per_level,
        min_d=args.min_d,
        rng=rng,
    )
    print(f"  Stratified selection: {len(selected)} items")

    # Phase 2: E-optimal fill
    if not args.no_fill and len(selected) < args.target_n:
        print(f"Phase 2: E-optimal fill to reach {args.target_n} items...")
        selected = eoptimal_fill(D_full, selected, args.target_n, n_caps, rng)
        print(f"  After fill: {len(selected)} items")
    elif len(selected) > args.target_n:
        print(f"  Stratified selection exceeds target ({len(selected)} > {args.target_n}); trimming...")
        # Keep items with highest D value for their primary capability
        d_primary = [D_full[j, D_full[j].argmax()] for j in selected]
        order = np.argsort(d_primary)[::-1]
        selected = [selected[i] for i in order[:args.target_n]]

    D_sel = D_full[selected]

    # Report
    selection_report(D_sel, D_full, cap_cols, short_names, alloc_df, args.target_n)

    # Save selected items CSV
    csv_path = Path(f"{args.output}_items.csv")
    out_df = df.iloc[selected][["dataset name", "sample id"] + cap_cols].copy()
    out_df.insert(2, "primary_capability",
                  [cap_cols[int(D_full[j].argmax())] for j in selected])
    out_df.insert(3, "primary_d_level",
                  [int(D_full[j].max()) for j in selected])
    out_df.to_csv(csv_path, index=False)
    print(f"\nSelected items saved to {csv_path}")

    # Plots
    print("\nGenerating plots...")

    plot_difficulty_spread_selected(
        D_sel, cap_cols, short_names, args.target_n,
        Path(f"{args.output}_difficulty_spread.png"), show,
    )

    plot_sv_comparison(
        D_sel, D_full, n_caps,
        Path(f"{args.output}_sv_comparison.png"), show,
        seed=args.seed,
    )

    print("Done.")


if __name__ == "__main__":
    main()
