#!/usr/bin/env python
"""
Generate suitability score plots for gpt-4o-mini across all domain-specific ability matrices.

Loads each inference result once and iterates through all domain matrices.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.core.model import collect_capability_means
from src.core.visualization import plot_suitability_scores
from src.utils.io import load_abilities, load_tasks, load_ability_matrix, load_agent_idata
from src.pipeline.suitability import score_all_tasks

# Configuration
AGENT = "gpt-4o-mini"
MODELS = {
    "add": "data/results/llm",
    "hierarchical": "data/results/llm_hierarchical",
    "softmin": "data/results/llm_softmin",
}
DOMAINS = {
    "AOP": "Admin, Org & Planning",
    "CMH": "Customer Service, Marketing & HR",
    "HSC": "Hospitality, Sales & Client Care",
    "MMR": "Manufacture, Maintenance & Repair",
    "NDP": "Numerical, Data & Programming",
    "WL": "Warehouse & Logistics",
}
DOMAIN_MATRIX_PATTERN = "data/processed/ability_matrix_combined_domain_{}.csv"

LABEL_MAP = {"gpt-4o-mini": "GPT-4o-mini"}


def main():
    print("Loading config...")
    abilities_df = load_abilities("config/abilities.csv")
    tasks_df = load_tasks("config/tasks.csv")
    capability_cols = list(abilities_df["Abilities"])

    # Load all domain matrices
    print("Loading domain ability matrices...")
    domain_matrices = {}
    for acronym, name in DOMAINS.items():
        path = DOMAIN_MATRIX_PATTERN.format(acronym)
        if Path(path).exists():
            domain_matrices[acronym] = load_ability_matrix(
                path, tasks_df, abilities_df, use_short_names=True
            )
            print(f"  - {acronym}: {name} ({domain_matrices[acronym].shape})")
        else:
            print(f"  - WARNING: {path} not found, skipping")

    # Process each model
    for model_name, idata_base in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Loading model: {model_name} ({idata_base})")
        print(f"{'='*60}")

        agent_idata, model_capability_cols = load_agent_idata(idata_base, [AGENT])
        if not agent_idata:
            print(f"  ERROR: No data for {AGENT}, skipping")
            continue

        if model_capability_cols is None:
            from src.utils.io import load_annotations
            _, model_capability_cols, _ = load_annotations("data/processed/annotations.csv")

        # Generate suitability for each domain
        for acronym, demand_df in domain_matrices.items():
            domain_name = DOMAINS[acronym]
            print(f"\n  Computing suitability for domain: {acronym} ({domain_name})")

            try:
                mean_df, ci_lo_df, ci_hi_df, samples = score_all_tasks(
                    agent_idata=agent_idata,
                    capability_cols=capability_cols,
                    demand_df=demand_df,
                    draws_cap=2000,
                    use_ratio=False,
                    weight_uncertainty="dirichlet",
                    kappa=300.0,
                    power_param=1.5,
                    demand_sharpness=3.0,
                    seed=123,
                )

                output_path = f"figures/suitability_{model_name}_{acronym}.png"
                plot_suitability_scores(
                    mean_df,
                    ci_lo_df,
                    ci_hi_df,
                    title=f"GPT-4o-mini Suitability ({model_name}) - {domain_name}",
                    save_path=output_path,
                    label_map=LABEL_MAP,
                )
                plt.close("all")

                print(f"    Scores: {mean_df[AGENT].min():.2f} - {mean_df[AGENT].max():.2f}")

            except Exception as e:
                print(f"    ERROR: {e}")

        # Free memory
        del agent_idata

    # Generate a comparison summary plot across domains for each model
    print(f"\n{'='*60}")
    print("Generating domain comparison summary plots...")
    print(f"{'='*60}")

    for model_name, idata_base in MODELS.items():
        print(f"\nLoading model: {model_name}")
        agent_idata, model_capability_cols = load_agent_idata(idata_base, [AGENT])
        if not agent_idata:
            continue

        if model_capability_cols is None:
            from src.utils.io import load_annotations
            _, model_capability_cols, _ = load_annotations("data/processed/annotations.csv")

        # Collect mean suitability per domain
        domain_means = {}
        domain_ci_lo = {}
        domain_ci_hi = {}

        for acronym, demand_df in domain_matrices.items():
            try:
                mean_df, ci_lo_df, ci_hi_df, _ = score_all_tasks(
                    agent_idata=agent_idata,
                    capability_cols=capability_cols,
                    demand_df=demand_df,
                    draws_cap=2000,
                    use_ratio=False,
                    weight_uncertainty="dirichlet",
                    kappa=300.0,
                    power_param=1.5,
                    demand_sharpness=3.0,
                    seed=123,
                )
                domain_means[acronym] = mean_df[AGENT].mean()
                domain_ci_lo[acronym] = ci_lo_df[AGENT].mean()
                domain_ci_hi[acronym] = ci_hi_df[AGENT].mean()
            except Exception as e:
                print(f"  ERROR for {acronym}: {e}")

        del agent_idata

        if not domain_means:
            continue

        # Plot domain comparison
        fig, ax = plt.subplots(figsize=(10, 5))
        domains_sorted = sorted(domain_means.keys(), key=lambda d: domain_means[d], reverse=True)
        x = np.arange(len(domains_sorted))
        means = [domain_means[d] for d in domains_sorted]
        lo = [domain_ci_lo[d] for d in domains_sorted]
        hi = [domain_ci_hi[d] for d in domains_sorted]
        err_lo = [m - l for m, l in zip(means, lo)]
        err_hi = [h - m for m, h in zip(means, hi)]
        labels = [f"{d}\n{DOMAINS[d]}" for d in domains_sorted]

        ax.barh(x, means, xerr=[err_lo, err_hi], capsize=4, color="steelblue", alpha=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("Mean Suitability Score (across tasks)")
        ax.set_title(
            f"GPT-4o-mini Domain Suitability ({model_name} pooling)",
            fontsize=14,
            weight="bold",
        )
        ax.grid(alpha=0.3, axis="x")
        ax.invert_yaxis()
        plt.tight_layout()

        output_path = f"figures/domain_comparison_{model_name}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved domain comparison: {output_path}")
        plt.close("all")

    print("\nDone!")


if __name__ == "__main__":
    main()
