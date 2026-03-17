#!/usr/bin/env python
"""
Combined simulation and inference script.

Simulates agent performance and immediately runs inference using the same model
parameters, ensuring the generative model matches the inference model exactly.

Usage:
    # Normalized additive (recommended)
    python scripts/run_sim_inference.py --pool add --normalize \
        --output data/results/sim_normadd

    # Softmin
    python scripts/run_sim_inference.py --pool softmin --tau 1.0 --mu-c 4.0 \
        --output data/results/sim_softmin

    # Single agent, save performance data
    python scripts/run_sim_inference.py --pool add --normalize \
        --agents social_specialist \
        --save-performance \
        --output data/results/sim_test
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning, module="pytensor")

import os, io
_old_stderr = sys.stderr
sys.stderr = io.StringIO()
import pytensor  # noqa: F401 — triggers the cl probe; suppress its stderr output
sys.stderr = _old_stderr
del _old_stderr

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from src.core.capabilities import validate_capability_alignment
from src.utils.io import load_abilities, load_annotations
from src.pipeline.simulation import create_simulated_data
from src.pipeline.inference import run_inference_batch


def print_recovery_table(C_df: pd.DataFrame, results: dict, capability_cols: list):
    """Print true vs. posterior-mean capability estimates for each agent."""
    print("\n--- Recovery Summary ---")
    all_errors = []

    for agent in C_df.index:
        if agent not in results:
            continue

        idata, _ = results[agent]
        est = idata.posterior["c"].mean(dim=("chain", "draw")).values
        true_vals = C_df.loc[agent, capability_cols].values
        errors = est - true_vals
        all_errors.extend(np.abs(errors).tolist())

        # Find specialist capabilities (those above the baseline level)
        baseline = true_vals.min()
        specialist_idx = np.where(true_vals > baseline)[0]

        print(f"\n  {agent}  (baseline c={baseline:.1f})")

        if len(specialist_idx) > 0:
            print(f"    {'Capability':<35} {'True':>6} {'Est':>6} {'Error':>7}")
            print(f"    {'-'*55}")
            for k in specialist_idx:
                cap = capability_cols[k]
                print(f"    {cap[:33]:<35} {true_vals[k]:6.1f} {est[k]:6.2f} {errors[k]:+7.2f}")
            # Baseline summary
            base_mask = true_vals == baseline
            base_errors = errors[base_mask]
            print(f"    {'[baseline capabilities]':<35} {baseline:6.1f} {est[base_mask].mean():6.2f}  mae={np.abs(base_errors).mean():.2f}")
        else:
            # Generalist: just show overall stats
            print(f"    Overall  mae={np.abs(errors).mean():.3f}  bias={errors.mean():.3f}  "
                  f"est_range=[{est.min():.2f}, {est.max():.2f}]")

    print(f"\n  Overall MAE across all agents/capabilities: {np.mean(all_errors):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate agent performance and run inference with identical model parameters"
    )

    # --- Data paths ---
    parser.add_argument("--abilities", default="config/abilities.csv",
                        help="Path to abilities definition file")
    parser.add_argument("--annotations", default="data/processed/annotations.csv",
                        help="Path to item annotations file")
    parser.add_argument("--output", required=True,
                        help="Output base path for inference results (e.g. data/results/sim_normadd)")

    # --- Model parameters: shared between simulation and inference ---
    model_group = parser.add_argument_group(
        "model parameters (applied to both simulation and inference)"
    )
    model_group.add_argument("--pool", choices=["add", "softmin"], default="add",
                             help="Pooling method: 'add' (compensatory) or 'softmin' (weakest-link)")
    model_group.add_argument("--tau", type=float, default=1.0,
                             help="Temperature for softmin pooling (higher = stricter weakest-link)")
    model_group.add_argument("--normalize", action="store_true", default=False,
                             help="Normalize additive pooling by active capabilities (mean instead of sum)")
    model_group.add_argument("--lam", type=float, default=1.0,
                             help="Per-level log step parameter")

    # --- Inference-only parameters ---
    inf_group = parser.add_argument_group("inference parameters")
    inf_group.add_argument("--mu-c", type=float, default=3.0,
                           help="Prior mean for capability levels (use 4.0-5.0 for softmin)")
    inf_group.add_argument("--no-hierarchical", action="store_true", default=False,
                           help="Disable hierarchical prior (enabled by default)")
    inf_group.add_argument("--no-fix-kappa", action="store_true", default=False,
                           help="Estimate kappa freely (fixed to 1 by default)")
    inf_group.add_argument("--no-coverage-aware", action="store_true", default=False,
                           help="Disable coverage-aware priors (enabled by default)")

    # --- Misc ---
    parser.add_argument("--agents", nargs="+", default=None,
                        help="Subset of agents to simulate and fit (default: all five)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for simulation (inference seed = seed + 100)")
    parser.add_argument("--save-performance", action="store_true", default=False,
                        help="Also save simulated performance CSVs alongside inference results")

    args = parser.parse_args()

    # --- Load data ---
    print("Loading data...")
    abilities_df = load_abilities(args.abilities)
    annotations_df, capability_cols, D = load_annotations(args.annotations)
    validate_capability_alignment(capability_cols, abilities_df, context="annotations vs abilities")
    print(f"  {D.shape[0]} items, {D.shape[1]} capabilities")

    # --- Echo model configuration ---
    pool_desc = args.pool + (" (normalized)" if args.normalize else "")
    if args.pool == "softmin":
        pool_desc += f", tau={args.tau}"
    print(f"\nModel configuration")
    print(f"  pool     : {pool_desc}")
    print(f"  lam      : {args.lam}")
    print(f"  mu_c     : {args.mu_c}  [inference prior only]")
    print(f"  Note: simulation uses alpha=0, kappa=1 (fixed); inference uses the priors above")

    # --- Simulate ---
    print("\nSimulating agent performance...")
    C_df, Ym_df, Pm_df = create_simulated_data(
        capability_cols=capability_cols,
        D=D,
        lam=args.lam,
        pool=args.pool,
        tau=args.tau,
        normalize=args.normalize,
        seed=args.seed,
    )

    agents = args.agents if args.agents else list(C_df.index)
    C_df = C_df.loc[agents]
    Ym_df = Ym_df.loc[agents]
    Pm_df = Pm_df.loc[agents]

    print(f"  Agents: {agents}")
    for agent in agents:
        avg = Ym_df.loc[agent].mean()
        print(f"    {agent:<28} avg success rate: {avg:.3f}")

    # --- Save performance data (optional) ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_performance:
        C_df.to_csv(f"{args.output}_profiles.csv")
        Ym_df.to_csv(f"{args.output}_performance.csv")
        Pm_df.to_csv(f"{args.output}_probabilities.csv")
        print(f"\n  Saved simulated data to {args.output}_{{profiles,performance,probabilities}}.csv")

    # --- Infer ---
    print(f"\nRunning inference (seed={args.seed + 100})...")
    results = run_inference_batch(
        agents=agents,
        performance_df=Ym_df,
        demand_matrix=D,
        item_index=Ym_df.columns.values,
        output_base_path=args.output,
        lam=args.lam,
        base_seed=args.seed + 100,
        hierarchical=not args.no_hierarchical,
        fix_kappa=not args.no_fix_kappa,
        coverage_aware=not args.no_coverage_aware,
        pool=args.pool,
        tau=args.tau,
        normalize=args.normalize,
        mu_c=args.mu_c,
        capability_cols=capability_cols,
    )

    # --- Recovery summary ---
    print_recovery_table(C_df, results, capability_cols)

    print(f"\nInference results saved to {args.output}_{{agent}}.nc")
    print("Done!")


if __name__ == "__main__":
    main()
