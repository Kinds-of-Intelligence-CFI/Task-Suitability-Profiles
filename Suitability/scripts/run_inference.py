#!/usr/bin/env python
"""
Run Bayesian inference to estimate capability profiles.

This script fits the capability model to performance data and saves
posterior samples for each agent.

Usage:
    # For simulated data
    python scripts/run_inference.py --mode simulated \
        --performance data/processed/simulated_performance.csv \
        --annotations data/processed/annotations.csv \
        --output data/results/simulated

    # For LLM evaluation data
    python scripts/run_inference.py --mode llm \
        --results data/raw/gpt-4o-mini_results.csv \
        --annotations data/processed/annotations.csv \
        --agent-name gpt-4o-mini \
        --output data/results/llm
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from Suitability.src.core.capabilities import standardize_capability_names, validate_capability_alignment, resolve_capability_filter
from Suitability.src.utils.io import load_abilities, load_annotations, load_performance_data
from Suitability.src.pipeline.inference import run_inference, run_inference_batch


def main():
    parser = argparse.ArgumentParser(description="Run capability inference")
    parser.add_argument(
        "--mode",
        choices=["simulated", "llm"],
        required=True,
        help="Data mode: 'simulated' or 'llm'",
    )
    parser.add_argument(
        "--abilities",
        default="config/abilities.csv",
        help="Path to abilities definition file",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to item annotations file",
    )
    parser.add_argument(
        "--performance",
        help="Path to simulated performance CSV (for mode=simulated)",
    )
    parser.add_argument(
        "--results",
        help="Path to LLM results CSV (for mode=llm)",
    )
    parser.add_argument(
        "--agent-name",
        default="agent",
        help="Agent name for LLM mode",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        help="List of agents to fit (for mode=simulated, default: all)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output base path for inference results",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Per-level log step parameter",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Base random seed",
    )
    parser.add_argument(
        "--no-hierarchical",
        action="store_true",
        default=False,
        help="Disable hierarchical prior (enabled by default)",
    )
    parser.add_argument(
        "--no-fix-kappa",
        action="store_true",
        default=False,
        help="Estimate kappa freely instead of fixing to 1 (fixed by default)",
    )
    parser.add_argument(
        "--no-coverage-aware",
        action="store_true",
        default=False,
        help="Disable coverage-aware priors (enabled by default)",
    )
    parser.add_argument(
        "--pool",
        choices=["add", "geom", "softmin"],
        default="add",
        help="Pooling method: 'add' (compensatory), 'geom' (soft-max), 'softmin' (non-compensatory weakest-link)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=1.0,
        help="Temperature for softmin pooling (higher = stricter weakest-link constraint)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize additive pooling by number of active capabilities (mean instead of sum). "
             "Improves stability across capability subsets.",
    )
    parser.add_argument(
        "--mu-c",
        type=float,
        default=3.0,
        help="Prior mean for capability levels (increase for softmin, e.g., 4.0-5.0)",
    )
    parser.add_argument(
        "--capabilities",
        nargs="+",
        default=None,
        help="Subset of capabilities to include (acronyms or full names, e.g., EM SM WM CF). Default: all",
    )

    args = parser.parse_args()

    print("Loading data...")
    abilities_df = load_abilities(args.abilities)

    if args.capabilities:
        resolved_filter = resolve_capability_filter(args.capabilities, abilities_df)
        print(f"  - Capability filter active: {args.capabilities} -> {resolved_filter}")
    else:
        resolved_filter = None

    if args.mode == "simulated":
        if not args.performance:
            parser.error("--performance is required for mode=simulated")

        annotations_df, capability_cols, D = load_annotations(
            args.annotations,
            capability_filter=resolved_filter,
            abilities_df=abilities_df,
        )
        performance_df = pd.read_csv(args.performance, index_col=0)
        item_index = performance_df.columns.values

        agents = args.agents if args.agents else list(performance_df.index)

    elif args.mode == "llm":
        if not args.results:
            parser.error("--results is required for mode=llm")

        performance_df, D, item_index, capability_cols = load_performance_data(
            results_path=args.results,
            annotations_path=args.annotations,
            agent_name=args.agent_name,
            capability_filter=resolved_filter,
            abilities_df=abilities_df,
        )
        agents = [args.agent_name]

    print(f"  - {D.shape[0]} items, {D.shape[1]} capability dimensions")
    print(f"  - Agents to fit: {agents}")

    # Validate alignment (skip when using a capability subset)
    if args.capabilities:
        print(f"  - Skipping full alignment check (capability subset active: {len(capability_cols)} of 18)")
    else:
        validate_capability_alignment(
            capability_cols, abilities_df, context="data vs abilities"
        )
        print("  - Capability alignment validated")

    # Calculate average performance
    print("\nPerformance summary:")
    for agent in agents:
        avg = performance_df.loc[agent].mean()
        total = performance_df.loc[agent].sum()
        print(f"  {agent}: {avg:.3f} ({int(total)}/{len(item_index)} correct)")

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Run inference
    print(f"\nRunning inference...")
    print(f"  - Capability order: {capability_cols[:3]}... ({len(capability_cols)} total)")
    results = run_inference_batch(
        agents=agents,
        performance_df=performance_df,
        demand_matrix=D,
        item_index=item_index,
        output_base_path=args.output,
        lam=args.lam,
        base_seed=args.seed,
        hierarchical=not args.no_hierarchical,
        fix_kappa=not args.no_fix_kappa,
        coverage_aware=not args.no_coverage_aware,
        pool=args.pool,
        tau=args.tau,
        normalize=args.normalize,
        mu_c=args.mu_c,
        capability_cols=capability_cols,
    )

    print(f"\nResults saved to {args.output}_*.nc")
    print("Done!")


if __name__ == "__main__":
    main()
