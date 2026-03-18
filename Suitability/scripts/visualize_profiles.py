#!/usr/bin/env python
"""
Visualize capability profiles from inference results.

This script generates radar plots, ICC curves, and forest plots
for analyzing inferred capability profiles.

Usage:
    python scripts/visualize_profiles.py \
        --agents gpt-4o-mini \
        --idata-base data/results/llm \
        --output figures/
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from Suitability.src.core.capabilities import standardize_capability_names
from Suitability.src.core.model import collect_capability_means, compute_capability_coverage
from Suitability.src.core.visualization import (
    plot_radar_capabilities,
    plot_demand_distribution,
    plot_icc_curve,
    plot_forest,
)
from Suitability.src.utils.io import load_abilities, load_annotations, load_agent_idata


def main():
    parser = argparse.ArgumentParser(description="Visualize capability profiles")
    parser.add_argument(
        "--agents",
        nargs="+",
        required=True,
        help="List of agent names to visualize",
    )
    parser.add_argument(
        "--idata-base",
        required=True,
        help="Base path for inference data files",
    )
    parser.add_argument(
        "--abilities",
        default="config/abilities.csv",
        help="Path to abilities definition file",
    )
    parser.add_argument(
        "--annotations",
        help="Path to annotations file (required for ICC plots, used for coverage in forest plots)",
    )
    parser.add_argument(
        "--output",
        default="figures/",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--plot-icc",
        action="store_true",
        help="Generate ICC plots for each capability",
    )
    parser.add_argument(
        "--plot-forest",
        action="store_true",
        help="Generate forest plots",
    )
    parser.add_argument(
        "--raw-scale",
        action="store_true",
        help="Use raw c values (log scale) for radar plots instead of theta. Not recommended as c can be negative.",
    )
    parser.add_argument(
        "--plot-demand-dist",
        action="store_true",
        help="Generate demand level distribution plot (requires --annotations)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save figures to disk without displaying them interactively",
    )

    args = parser.parse_args()

    print("Loading data...")
    abilities_df = load_abilities(args.abilities)
    display_capability_cols = list(abilities_df["Abilities"])  # Canonical display order

    print(f"  - {len(abilities_df)} abilities")
    print(f"  - Agents: {args.agents}")

    print("\nLoading inference data...")
    agent_idata, model_capability_cols = load_agent_idata(args.idata_base, args.agents)

    if not agent_idata:
        print("ERROR: No inference data found for specified agents")
        sys.exit(1)

    print(f"  - Loaded data for {list(agent_idata.keys())}")

    # If no metadata found, try to load capability order from annotations
    if model_capability_cols is None:
        if args.annotations:
            print("  - No capability metadata found, loading order from annotations...")
            _, model_capability_cols, _ = load_annotations(args.annotations)
        else:
            print("  - WARNING: No capability metadata found and no --annotations provided.")
            print("    Assuming model order matches abilities.csv order (may be incorrect!)")
            model_capability_cols = display_capability_cols

    print(f"  - Model capability order: {model_capability_cols[:3]}... ({len(model_capability_cols)} total)")

    # When a capability subset was used, restrict display order to those in the model
    model_set = set(model_capability_cols)
    display_capability_cols = [c for c in display_capability_cols if c in model_set]

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract capability means in log scale (reordering from model order to display order)
    # The radar plot will transform to theta scale internally
    capability_df = collect_capability_means(
        agent_idata,
        model_capability_cols=model_capability_cols,
        display_capability_cols=display_capability_cols,
        use_theta=False,  # Keep as log scale; radar plot transforms to theta internally
    )

    # Label map for nicer display names
    label_map = {
        "strong_generalist": "Strong Generalist",
        "weak_generalist": "Weak Generalist",
        "social_specialist": "Social Specialist",
        "strategic_specialist": "Strategic Specialist",
        "physical_specialist": "Physical Specialist",
        "gpt-4o-mini": "GPT-4o-mini",
    }

    # Generate radar plot (uses theta scale by default for always-positive values)
    print("\nGenerating radar plot...")
    plot_radar_capabilities(
        capability_df,
        capability_info=abilities_df,
        overlay=True,
        save_path=str(output_dir / "capability_radar.png"),
        label_map=label_map,
        use_theta=not args.raw_scale,  # Default True; use theta for always-positive values
        show=not args.no_show,
    )

    # Load annotations if needed for demand distribution or ICC plots
    # Filter to model capabilities so D columns align with the model's subset
    D = None
    annotations_cap_cols = None
    if args.annotations and (args.plot_icc or args.plot_demand_dist):
        _, annotations_cap_cols, D = load_annotations(
            args.annotations,
            capability_filter=model_capability_cols,
        )

    # Generate demand distribution plot
    if args.plot_demand_dist:
        if D is None:
            print("WARNING: --annotations required for demand distribution plot, skipping")
        else:
            print("\nGenerating demand distribution plot...")
            counts_df = plot_demand_distribution(
                D,
                annotations_cap_cols,
                abilities_df=abilities_df,
                save_path=str(output_dir / "demand_distribution.png"),
                show=not args.no_show,
            )
            print("\nDemand level counts:")
            print(counts_df.to_string())

    # Generate forest plots
    if args.plot_forest:
        print("\nGenerating forest plots...")

        # Compute coverage using model capability order
        coverage = None
        if args.annotations:
            # Load D filtered to model capabilities for correct coverage mapping
            _, _, D_full = load_annotations(
                args.annotations,
                capability_filter=model_capability_cols,
            )
            coverage = compute_capability_coverage(D_full)
            print(f"  - Coverage range: {coverage.min()*100:.1f}% - {coverage.max()*100:.1f}%")

        for agent in agent_idata:
            # Include kappa in forest plot when it was estimated (not fixed)
            forest_vars = ["c"]
            if "kappa" in agent_idata[agent].posterior:
                forest_vars.append("kappa")
            plot_forest(
                agent_idata[agent],
                var_names=forest_vars,
                save_path=str(output_dir / f"forest_{agent}.png"),
                abilities_df=abilities_df,
                coverage=coverage,
                capability_cols=model_capability_cols,
                show=not args.no_show,
            )

    # Generate ICC plots
    if args.plot_icc:
        if D is None:
            print("WARNING: --annotations required for ICC plots, skipping")
        else:
            print("\nGenerating ICC plots...")
            for agent in agent_idata:
                for k, cap_name in enumerate(annotations_cap_cols):
                    plot_icc_curve(
                        agent_idata[agent],
                        annotations_cap_cols,
                        D,
                        k=k,
                        save_path=str(output_dir / f"icc_{agent}_{cap_name}.png"),
                        show=not args.no_show,
                    )

    # Print capability summary
    print("\nCapability Profile Summary (posterior means):")
    print(capability_df.round(2).to_string())

    print("\nDone!")


if __name__ == "__main__":
    main()
