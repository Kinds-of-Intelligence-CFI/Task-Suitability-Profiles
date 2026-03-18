#!/usr/bin/env python
"""
Build ability matrices from questionnaire data.

This script processes questionnaire responses to create ability matrices
that can be used for computing suitability scores. Supports filtering
by data source (companies/online) and domain.

Usage:
    # Build matrix from all data
    python scripts/build_ability_matrix.py \
        --companies questionnaire/Future_of_skills_companies_20260112.csv \
        --online questionnaire/Future_of_skills_online_20260112.csv \
        --output data/processed/ability_matrix_all.csv

    # Build matrix for specific domain
    python scripts/build_ability_matrix.py \
        --companies questionnaire/Future_of_skills_companies_20260112.csv \
        --online questionnaire/Future_of_skills_online_20260112.csv \
        --domain 3 \
        --output data/processed/ability_matrix_NDP.csv

    # Build matrix from online participants only
    python scripts/build_ability_matrix.py \
        --online questionnaire/Future_of_skills_online_20260112.csv \
        --source online \
        --output data/processed/ability_matrix_online.csv

    # Build matrices for all domains
    python scripts/build_ability_matrix.py \
        --companies questionnaire/Future_of_skills_companies_20260112.csv \
        --online questionnaire/Future_of_skills_online_20260112.csv \
        --all-domains \
        --output data/processed/ability_matrix
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from Suitability.src.pipeline.questionnaire import (
    load_questionnaire_data,
    apply_quality_control,
    build_ability_matrix,
    build_ability_matrices_by_domain,
    build_ability_matrices_by_source,
    compare_ability_matrices,
    get_domain_label,
    get_participant_stats,
    DOMAIN_INFO,
)
from Suitability.src.utils.io import save_ability_matrix


def plot_ability_matrix(
    matrix: pd.DataFrame,
    title: str,
    save_path: str,
) -> None:
    """Plot ability matrix as a heatmap."""
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="Blues",
        cbar_kws={"label": "Weighted Importance"},
    )
    plt.xlabel("Capability")
    plt.ylabel("Work Activity")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build ability matrices from questionnaire data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data source arguments
    parser.add_argument(
        "--companies",
        help="Path to company-recruited participants CSV",
    )
    parser.add_argument(
        "--online",
        help="Path to online-recruited participants CSV",
    )

    # Reference data
    parser.add_argument(
        "--tasks",
        default="config/tasks.csv",
        help="Path to tasks definition CSV",
    )
    parser.add_argument(
        "--abilities",
        default="config/abilities.csv",
        help="Path to abilities definition CSV",
    )
    parser.add_argument(
        "--domains",
        default=None,
        help="Path to domains definition CSV (optional)",
    )

    # Filtering options
    parser.add_argument(
        "--source",
        choices=["companies", "online", "combined"],
        default="combined",
        help="Filter by data source (default: combined)",
    )
    parser.add_argument(
        "--domain",
        type=int,
        default=0,
        choices=range(7),
        help="Filter by domain (0=all, 1-6 for specific domains)",
    )
    parser.add_argument(
        "--all-domains",
        action="store_true",
        help="Build separate matrices for all domains",
    )
    parser.add_argument(
        "--compare-sources",
        action="store_true",
        help="Compare matrices between data sources",
    )

    # Quality control
    parser.add_argument(
        "--completion-threshold",
        type=float,
        default=100,
        help="Minimum completion percentage (default: 100)",
    )
    parser.add_argument(
        "--duration-threshold",
        type=float,
        default=600,
        help="Minimum duration in seconds (default: 600)",
    )
    parser.add_argument(
        "--quiz-threshold",
        type=float,
        default=50,
        help="Minimum quiz score percentage (default: 50)",
    )

    # Output options
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for ability matrix CSV (or base path for multiple matrices)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate heatmap plots",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize rows to sum to 100",
    )

    args = parser.parse_args()

    if not args.companies and not args.online:
        parser.error("At least one of --companies or --online must be provided")

    # Load data
    print("Loading questionnaire data...")
    df, tasks_df, abilities_df, domains_df = load_questionnaire_data(
        companies_path=args.companies,
        online_path=args.online,
        tasks_path=args.tasks,
        abilities_path=args.abilities,
        domains_path=args.domains,
    )

    # Apply quality control
    print("\nApplying quality control...")
    df = apply_quality_control(
        df,
        completion_threshold=args.completion_threshold,
        duration_threshold=args.duration_threshold,
        quiz_threshold=args.quiz_threshold,
    )

    output_path = Path(args.output)
    # Strip .csv extension if present (we'll add it back as needed)
    if output_path.suffix.lower() == ".csv":
        output_path = output_path.with_suffix("")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.all_domains:
        # Build matrices for all domains
        print("\nBuilding ability matrices for all domains...")
        matrices = build_ability_matrices_by_domain(
            df, tasks_df, abilities_df, source=args.source
        )

        for domain, matrix in matrices.items():
            domain_label = DOMAIN_INFO[domain]["acronym"]
            matrix_path = str(output_path) + f"_domain_{domain_label}"
            save_ability_matrix(matrix, matrix_path + ".csv")

            if args.plot:
                title = f"Ability Matrix - {get_domain_label(domain)}"
                plot_ability_matrix(matrix, title, matrix_path + ".png")

    elif args.compare_sources:
        # Compare matrices between sources
        print("\nBuilding and comparing ability matrices by source...")
        matrices = build_ability_matrices_by_source(
            df, tasks_df, abilities_df, domain=args.domain
        )

        if len(matrices) < 2:
            print("ERROR: Need at least 2 sources to compare")
            sys.exit(1)

        sources = list(matrices.keys())
        colwise, rowwise = compare_ability_matrices(
            matrices[sources[0]], matrices[sources[1]],
            name_a=sources[0], name_b=sources[1],
        )

        print("\nAbility-wise comparison (columns):")
        print(colwise.round(3).to_string())

        print("\nTask-wise comparison (rows):")
        print(rowwise.round(3).to_string())

        # Save individual matrices
        for source, matrix in matrices.items():
            matrix_path = str(output_path) + f"_{source}"
            save_ability_matrix(matrix, matrix_path + ".csv")

            if args.plot:
                title = f"Ability Matrix - {source.capitalize()} ({get_domain_label(args.domain)})"
                plot_ability_matrix(matrix, title, matrix_path + ".png")

        # Save comparison results
        comparison_path = str(output_path) + "_comparison"
        colwise.to_csv(comparison_path + "_abilities.csv")
        rowwise.to_csv(comparison_path + "_tasks.csv")
        print(f"\nSaved comparison results to {comparison_path}_*.csv")

    else:
        # Build single matrix
        print(f"\nBuilding ability matrix (source={args.source}, domain={args.domain})...")

        stats = get_participant_stats(df, source=args.source, domain=args.domain)
        print(f"  Participants: {stats['total']}")

        matrix = build_ability_matrix(
            df, tasks_df, abilities_df,
            source=args.source,
            domain=args.domain,
            normalize=args.normalize,
        )

        # Ensure output has .csv extension
        output_csv = str(output_path)
        if not output_csv.endswith(".csv"):
            output_csv += ".csv"

        save_ability_matrix(matrix, output_csv)

        if args.plot:
            title = f"Ability Matrix - {args.source.capitalize()} ({get_domain_label(args.domain)})"
            plot_path = output_csv.replace(".csv", ".png")
            plot_ability_matrix(matrix, title, plot_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
