#!/usr/bin/env python
"""
Compute suitability scores for agents across tasks.

This script computes how suitable each agent is for different tasks based
on their inferred capability profiles and task demand profiles.

Usage:
    python scripts/run_suitability.py \
        --agents strong_generalist weak_generalist social_specialist \
        --idata-base data/results/simulated \
        --ability-matrix data/processed/ability_matrix.csv \
        --output figures/suitability_scores.png
"""

import argparse
from pathlib import Path
import sys

import pandas as pd
import numpy as np

from Suitability.src.core.capabilities import standardize_capability_names, validate_capability_alignment
from Suitability.src.core.model import collect_capability_means
from Suitability.src.core.visualization import plot_radar_capabilities, plot_suitability_scores
from Suitability.src.utils.io import load_abilities, load_tasks, load_ability_matrix, load_agent_idata
from Suitability.src.pipeline.suitability import score_all_tasks


def main():
    parser = argparse.ArgumentParser(description="Compute suitability scores")
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="List of agent names to include",
    )
    parser.add_argument(
        "--idata-base",
        required=True,
        help="Base path for inference data files",
    )
    parser.add_argument(
        "--abilities",
        default=Path(__file__).parent.parent / "config" / "abilities.csv",
        help="Path to abilities definition file",
    )
    parser.add_argument(
        "--tasks",
        default=Path(__file__).parent.parent / "config" / "tasks.csv",
        help="Path to tasks definition file",
    )
    parser.add_argument(
        "--ability-matrix",
        required=True,
        help="Path to ability matrix CSV (task x ability demands)",
    )
    parser.add_argument(
        "--output",
        default=Path(__file__).parent.parent / "data" / "results" / "figures" / "suitability_scores.png",
        help="Output path for suitability plot",
    )
    parser.add_argument(
        "--radar-output",
        default=Path(__file__).parent.parent / "data" / "results" / "figures" / "capability_radar.png",
        help="Output path for radar plot",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=1.5,
        help="Power parameter for aggregation",
    )
    parser.add_argument(
        "--kappa",
        type=float,
        default=300.0,
        help="Dirichlet concentration parameter",
    )
    parser.add_argument(
        "--sharpness",
        type=float,
        default=3.0,
        help="Demand sharpness parameter",
    )
    parser.add_argument(
        "--column-normalize",
        action="store_true",
        help=(
            "Min-max normalise each capability column across tasks before weighting. "
            "Converts absolute demand levels into relative demand, improving score "
            "variation when the raw demand matrix is near-uniform across tasks."
        ),
    )
    parser.add_argument(
        "--use-ratio",
        action="store_true",
        help="Use ratio-scale capabilities (theta = exp(c)) instead of log-scale. Required for negative power values.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display plots interactively (just save to file)",
    )

    args = parser.parse_args()

    print("Loading data...")
    abilities_df = load_abilities(args.abilities)
    tasks_df = load_tasks(args.tasks)
    demand_df = load_ability_matrix(args.ability_matrix, tasks_df, abilities_df, use_short_names=True)

    capability_cols = list(abilities_df["Abilities"])

    print(f"  - {len(abilities_df)} abilities")
    print(f"  - {len(tasks_df)} tasks")
    print(f"  - Agents: {args.agents}")

    print("\nLoading inference data...")
    agent_idata, model_capability_cols = load_agent_idata(args.idata_base, args.agents)

    if not agent_idata:
        print("ERROR: No inference data found for specified agents")
        sys.exit(1)

    print(f"  - Loaded data for {list(agent_idata.keys())}")

    # model_capability_cols is the subset actually inferred (may be < 18)
    # fall back to all 18 only if no metadata was found
    scoring_cols = model_capability_cols if model_capability_cols else capability_cols
    print(f"  - Scoring over {len(scoring_cols)} capabilities")

    # Extract capability means (model_capability_cols from metadata, capability_cols for display)
    capability_df = collect_capability_means(
        agent_idata,
        scoring_cols,
        capability_cols,
    )

    # Generate radar plot (log scale by default)
    print("\nGenerating capability radar plot...")
    plot_radar_capabilities(
        capability_df,
        capability_info=abilities_df,
        overlay=True,
        title="Agent Capability Profiles (log scale)",
        save_path=args.radar_output,
        use_theta=False,
        show=not args.no_show,
    )

    # Compute suitability scores using only the inferred capability subset.
    # demand_weights are re-normalized over the subset automatically.
    print("\nComputing suitability scores...")
    mean_df, ci_lo_df, ci_hi_df, samples = score_all_tasks(
        agent_idata=agent_idata,
        capability_cols=scoring_cols,
        demand_df=demand_df,
        draws_cap=2000,
        use_ratio=args.use_ratio,
        weight_uncertainty="dirichlet",
        kappa=args.kappa,
        power_param=args.power,
        demand_sharpness=args.sharpness,
        column_normalize=args.column_normalize,
        seed=123,
    )

    # Generate suitability plot
    print("\nGenerating suitability scores plot...")
    plot_suitability_scores(
        mean_df,
        ci_lo_df,
        ci_hi_df,
        title="Agent Suitability Scores by Task",
        save_path=args.output,
        show=not args.no_show,
    )

    # Print summary
    print("\nSuitability Score Summary:")
    print(mean_df.round(2).to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
