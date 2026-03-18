#!/usr/bin/env python
"""
Run simulation of agent performance.

This script simulates agent performance on benchmark items based on predefined
capability profiles. Useful for testing and validating the inference pipeline.

Usage:
    python scripts/run_simulation.py --output data/processed/simulated
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np

from Suitability.src.core.capabilities import standardize_capability_names, validate_capability_alignment
from Suitability.src.utils.io import load_abilities, load_annotations
from Suitability.src.pipeline.simulation import create_simulated_data


def main():
    parser = argparse.ArgumentParser(description="Simulate agent performance data")
    parser.add_argument(
        "--abilities",
        default="config/abilities.csv",
        help="Path to abilities definition file",
    )
    parser.add_argument(
        "--annotations",
        default="data/processed/annotations.csv",
        help="Path to item annotations file",
    )
    parser.add_argument(
        "--output",
        default="data/processed/simulated",
        help="Output base path for simulated data",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=1.0,
        help="Per-level log step parameter",
    )
    parser.add_argument(
        "--pool",
        choices=["add", "softmin"],
        default="add",
        help="Pooling method for generating simulated data: 'add' (compensatory) or 'softmin' (weakest-link)",
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
        help="Normalize additive pooling by number of active capabilities (mean instead of sum)",
    )

    args = parser.parse_args()

    print("Loading data...")
    abilities_df = load_abilities(args.abilities)
    annotations_df, capability_cols, D = load_annotations(args.annotations)

    print(f"  - {len(abilities_df)} abilities")
    print(f"  - {D.shape[0]} items, {D.shape[1]} capability dimensions")

    # Validate alignment
    validate_capability_alignment(
        capability_cols, abilities_df, context="annotations vs abilities"
    )
    print("  - Capability alignment validated")

    print(f"\nSimulating agent performance (pool={args.pool}, tau={args.tau}, normalize={args.normalize})...")
    C_df, Ym_df, Pm_df = create_simulated_data(
        capability_cols=capability_cols,
        D=D,
        lam=args.lam,
        pool=args.pool,
        tau=args.tau,
        normalize=args.normalize,
        seed=args.seed,
    )

    print(f"  - Created {len(C_df)} agent profiles")
    print(f"  - Average performance by agent:")
    for agent in Ym_df.index:
        avg = Ym_df.loc[agent].mean()
        print(f"    {agent}: {avg:.3f}")

    # Save outputs
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    C_df.to_csv(f"{args.output}_profiles.csv")
    Ym_df.to_csv(f"{args.output}_performance.csv")
    Pm_df.to_csv(f"{args.output}_probabilities.csv")

    print(f"\nSaved outputs to {args.output}_*.csv")
    print("Done!")


if __name__ == "__main__":
    main()
