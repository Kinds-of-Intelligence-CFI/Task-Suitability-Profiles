#!/usr/bin/env python
"""
Fit the hierarchical population IRT model to human participant data.

Estimates capability profiles for all participants jointly via partial pooling,
giving tighter individual estimates than fitting each person independently.

Outputs:
  - {output}_population.nc       : ArviZ InferenceData (population + individual posteriors)
  - {output}_population_meta.json: Participant names and capability column order

Usage:
    python scripts/run_population_inference.py \
        --performance data/processed/human_performance.csv \
        --annotations data/processed/selected_battery_items.csv \
        --output data/results/population \
        --pool add --normalize

    # With softmin pooling
    python scripts/run_population_inference.py \
        --performance data/processed/human_performance.csv \
        --annotations data/processed/selected_battery_items.csv \
        --output data/results/population_softmin \
        --pool softmin --tau 0.25 --mu-c 2.5

Performance CSV format:
    - Rows   : one per participant (index = participant ID)
    - Columns: one per item, named as "{dataset_name}_{sample_id}"
    - Values : 0 (incorrect), 1 (correct), or NaN (not attempted)
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import arviz as az

from Suitability.src.utils.io import load_annotations, load_abilities, load_population_idata
from Suitability.src.pipeline.inference import run_population_inference
from Suitability.src.core.model import extract_population_capability_samples
from Suitability.src.core.visualization import plot_radar_capabilities
from Suitability.src.core.model import collect_capability_means


def load_human_performance(
    performance_path: str,
    annotations_path: str,
    capability_filter=None,
    abilities_df=None,
):
    """
    Load and align human performance data with annotations.

    Performance CSV has participants as rows, composite item IDs as columns.
    Annotations are aligned on composite_id = {dataset_name}_{sample_id}.
    """
    ann_df, capability_cols, _ = load_annotations(
        annotations_path,
        capability_filter=capability_filter,
        abilities_df=abilities_df,
    )
    ann_df["composite_id"] = (
        ann_df["dataset_name"].astype(str) + "_" + ann_df["sample_id"].astype(str)
    )
    ann_df = ann_df.set_index("composite_id")

    perf_df = pd.read_csv(performance_path, index_col=0)

    # Align items: keep only items present in both
    common_items = list(set(perf_df.columns) & set(ann_df.index))
    if not common_items:
        raise ValueError(
            "No common items between performance CSV and annotations. "
            "Ensure performance columns are formatted as '{dataset_name}_{sample_id}'."
        )
    missing = len(perf_df.columns) - len(common_items)
    if missing > 0:
        print(f"  WARNING: {missing} items in performance CSV not found in annotations (dropped)")

    perf_df = perf_df[common_items]
    D = ann_df.loc[common_items, capability_cols].to_numpy(float)
    item_index = np.array(common_items)

    print(f"  {len(perf_df)} participants, {len(common_items)} items, {len(capability_cols)} capabilities")
    n_missing = perf_df.isna().sum().sum()
    if n_missing > 0:
        print(f"  {n_missing} missing responses ({n_missing / perf_df.size * 100:.1f}%)")

    return perf_df, D, item_index, capability_cols


def main():
    parser = argparse.ArgumentParser(
        description="Fit hierarchical population IRT model to human participants"
    )
    parser.add_argument(
        "--performance",
        required=True,
        help="Path to human performance CSV (participants × items)",
    )
    parser.add_argument(
        "--annotations",
        default="data/processed/annotations.csv",
        help="Path to annotations CSV",
    )
    parser.add_argument(
        "--abilities",
        default="config/abilities.csv",
        help="Path to abilities definition CSV",
    )
    parser.add_argument(
        "--capabilities",
        nargs="+",
        default=None,
        help="Subset of capabilities to include (acronyms or full names)",
    )
    parser.add_argument(
        "--output",
        default="data/results/population",
        help="Output base path (default: data/results/population)",
    )
    parser.add_argument(
        "--pool",
        choices=["add", "geom", "softmin"],
        default="add",
        help="Pooling method (default: add)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize additive pooling by number of active capabilities",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature for softmin/geom pooling (default: 1.0)",
    )
    parser.add_argument(
        "--mu-c",
        type=float,
        default=2.5,
        help="Prior mean for population capabilities (default: 2.5)",
    )
    parser.add_argument(
        "--sigma-c",
        type=float,
        default=0.8,
        help="Prior SD for population capability means (default: 0.8)",
    )
    parser.add_argument(
        "--sigma-pop-sd",
        type=float,
        default=0.5,
        help="Scale for HalfNormal prior on within-population spread (default: 0.5)",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Per-level log step for difficulty scaling (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--radar-output",
        default=None,
        help="Optional path to save capability radar plot",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively",
    )

    args = parser.parse_args()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # Load abilities for acronym resolution
    abilities_df = load_abilities(args.abilities)
    cap_filter = args.capabilities  # None = all

    # Load and align data
    print("Loading data...")
    perf_df, D, item_index, capability_cols = load_human_performance(
        performance_path=args.performance,
        annotations_path=args.annotations,
        capability_filter=cap_filter,
        abilities_df=abilities_df,
    )

    # Run inference
    idata, model = run_population_inference(
        performance_df=perf_df,
        demand_matrix=D,
        item_index=item_index,
        capability_cols=capability_cols,
        output_base_path=args.output,
        lam=args.lam,
        seed=args.seed,
        save_results=True,
        pool=args.pool,
        tau=args.tau,
        normalize=args.normalize,
        mu_c=args.mu_c,
        sigma_c=args.sigma_c,
        sigma_pop_sd=args.sigma_pop_sd,
    )

    # Diagnostics summary
    print("\nDivergences:", idata.sample_stats.diverging.sum().item())
    r_hat = az.rhat(idata, var_names=["mu_pop", "sigma_pop", "c_offset"])
    max_rhat = float(r_hat.to_array().max())
    print(f"Max R-hat (mu_pop, sigma_pop, c_offset): {max_rhat:.3f}")

    # Population-level capability summary
    print("\nPopulation capability means (mu_pop):")
    mu_means = idata.posterior["mu_pop"].mean(dim=("chain", "draw")).to_numpy()
    mu_sd = idata.posterior["mu_pop"].std(dim=("chain", "draw")).to_numpy()
    for cap, mean, sd in sorted(
        zip(capability_cols, mu_means, mu_sd), key=lambda x: -x[1]
    ):
        print(f"  {cap:<45} {mean:+.3f} ± {sd:.3f}")

    # Optional radar plot of population mean vs individual means
    if args.radar_output:
        print("\nGenerating radar plot...")
        participant_idata = extract_population_capability_samples(
            idata,
            participant_names=list(perf_df.index),
            draws=2000,
        )
        cap_df = collect_capability_means(
            participant_idata,
            model_capability_cols=capability_cols,
        )
        plot_radar_capabilities(
            cap_df,
            capability_info=abilities_df,
            overlay=True,
            title="Population capability profiles",
            save_path=args.radar_output,
            use_theta=False,
            show=not args.no_show,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
