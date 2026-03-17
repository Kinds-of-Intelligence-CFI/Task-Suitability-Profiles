"""
Plot mean accuracy for each capability.

For each capability, computes the mean accuracy on items where that capability
has demand > 0.
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def compute_accuracy_by_capability(
    results_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute mean accuracy for each capability.

    Args:
        results_df: DataFrame with columns [dataset_name, sample_id, score]
        annotations_df: DataFrame with capability demand columns

    Returns:
        DataFrame with mean accuracy per capability
    """
    # Merge results with annotations on (dataset_name, sample_id)
    merged = results_df.merge(
        annotations_df,
        left_on=["dataset_name", "sample_id"],
        right_on=["dataset name", "sample id"],
        how="inner",
    )

    # Get capability columns (all columns except identifiers)
    id_cols = ["dataset_name", "sample_id", "score", "metadata", "task_name",
               "dataset name", "sample id"]
    capability_cols = [c for c in merged.columns if c not in id_cols]

    # Compute mean accuracy for each capability
    results = []
    for cap in capability_cols:
        # Filter items where this capability has demand > 0
        mask = merged[cap] > 0
        if mask.sum() > 0:
            mean_acc = merged.loc[mask, "score"].mean()
            n_items = mask.sum()
            results.append({
                "capability": cap,
                "mean_accuracy": mean_acc,
                "n_items": n_items,
            })

    return pd.DataFrame(results)


def plot_accuracy_by_capability(
    acc_df: pd.DataFrame,
    abilities_df: pd.DataFrame = None,
    figsize: tuple = (12, 6),
    title: str = "Mean Accuracy by Capability",
    save_path: str = None,
) -> None:
    """
    Plot a bar chart of mean accuracy per capability.
    """
    # Get acronym mapping if abilities_df provided
    if abilities_df is not None:
        acronym_map = dict(zip(abilities_df["Abilities"], abilities_df["Acronym"]))
        acc_df = acc_df.copy()
        acc_df["label"] = acc_df["capability"].map(
            lambda x: acronym_map.get(x, x[:8])
        )
    else:
        acc_df["label"] = acc_df["capability"].str[:8]

    # Sort by accuracy
    acc_df = acc_df.sort_values("mean_accuracy", ascending=True)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.RdYlGn(acc_df["mean_accuracy"].values)
    bars = ax.barh(acc_df["label"], acc_df["mean_accuracy"], color=colors)

    # Add value labels
    for bar, acc, n in zip(bars, acc_df["mean_accuracy"], acc_df["n_items"]):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{acc:.2f} (n={n})",
            va="center", fontsize=9,
        )

    ax.set_xlim(0, 1.15)
    ax.set_xlabel("Mean Accuracy")
    ax.set_ylabel("Capability")
    ax.set_title(title, fontsize=14, weight="bold")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot mean accuracy by capability")
    parser.add_argument("--results", required=True, help="Path to results CSV")
    parser.add_argument("--annotations", required=True, help="Path to annotations CSV")
    parser.add_argument("--abilities", default="config/abilities.csv", help="Path to abilities CSV")
    parser.add_argument("--output", default="figures/accuracy_by_capability.png", help="Output path")
    parser.add_argument("--title", default="Mean Accuracy by Capability", help="Plot title")
    args = parser.parse_args()

    # Load data
    results_df = pd.read_csv(args.results)
    annotations_df = pd.read_csv(args.annotations)
    abilities_df = pd.read_csv(args.abilities)

    # Compute accuracy
    acc_df = compute_accuracy_by_capability(results_df, annotations_df)

    # Plot
    plot_accuracy_by_capability(
        acc_df,
        abilities_df=abilities_df,
        title=args.title,
        save_path=args.output,
    )


if __name__ == "__main__":
    main()
