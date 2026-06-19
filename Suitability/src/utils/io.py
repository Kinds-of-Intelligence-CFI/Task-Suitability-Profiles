"""
Data loading and saving utilities.
"""

import os
import json
import pandas as pd
import numpy as np
import dill
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

# Lazy import for arviz (heavy dependency)
if TYPE_CHECKING:
    import arviz as az

from ..core.capabilities import standardize_capability_names, resolve_capability_filter


def _get_arviz():
    """Lazy import for arviz."""
    import arviz as az
    return az


def load_abilities(path: str) -> pd.DataFrame:
    """
    Load the abilities definition file.

    Args:
        path: Path to abilities.csv

    Returns:
        DataFrame with ability definitions
    """
    return pd.read_csv(path)


def load_tasks(path: str) -> pd.DataFrame:
    """
    Load the tasks definition file.

    Args:
        path: Path to tasks.csv

    Returns:
        DataFrame with task definitions
    """
    return pd.read_csv(path)


def load_annotations(
    path: str,
    standardize_names: bool = True,
    capability_filter: Optional[List[str]] = None,
    abilities_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str], np.ndarray]:
    """
    Load the capability annotation table for benchmark items.

    Args:
        path: Path to annotations CSV file
        standardize_names: If True, standardize capability column names
        capability_filter: Optional list of capability names or acronyms to keep.
                           Requires abilities_df for acronym resolution.
        abilities_df: DataFrame with 'Abilities' and 'Acronym' columns,
                      needed when capability_filter contains acronyms.

    Returns:
        Tuple of (DataFrame, capability_cols, demand_matrix D)
    """
    df = pd.read_csv(path)

    # Rename identifier columns if needed
    if "sample id" in df.columns:
        df = df.rename(columns={"sample id": "sample_id"})
    if "dataset name" in df.columns:
        df = df.rename(columns={"dataset name": "dataset_name"})

    # Get capability columns (everything after identifiers)
    id_cols = ["dataset_name", "sample_id"] if "dataset_name" in df.columns else []
    if not id_cols:
        id_cols = list(df.columns[:2])

    capability_cols_raw = [c for c in df.columns if c not in id_cols]

    if standardize_names:
        capability_cols = standardize_capability_names(capability_cols_raw)
        rename_map = dict(zip(capability_cols_raw, capability_cols))
        df = df.rename(columns=rename_map)
    else:
        capability_cols = capability_cols_raw

    # Apply capability filter if provided
    if capability_filter is not None:
        if abilities_df is not None:
            resolved = resolve_capability_filter(capability_filter, abilities_df)
        else:
            resolved = list(capability_filter)
        # Keep only resolved capabilities, preserving original column order
        capability_cols = [c for c in capability_cols if c in resolved]

    D = df[capability_cols].to_numpy(float)

    return df, capability_cols, D


def load_ability_matrix(
    path: str,
    tasks_df: pd.DataFrame,
    abilities_df: pd.DataFrame,
    use_short_names: bool = False,
) -> pd.DataFrame:
    """
    Load and format the ability matrix (task x ability demands).

    Args:
        path: Path to ability_matrix CSV
        tasks_df: Tasks definition DataFrame
        abilities_df: Abilities definition DataFrame
        use_short_names: If True, use "Heading" column for task names; else use "Task"

    Returns:
        DataFrame with tasks as index, abilities as columns, demands as values
    """
    df = pd.read_csv(path)

    # Map task codes to names
    task_col = "Heading" if use_short_names else "Task"
    task_mapping = dict(zip(df.iloc[:, 0], tasks_df[task_col]))

    # Map acronyms to full ability names
    capability_mapping = dict(zip(abilities_df["Acronym"], abilities_df["Abilities"]))

    # Set index and rename columns
    df = df.set_index(df.columns[0])
    df = df.rename(columns=capability_mapping)
    df.index = df.index.map(lambda x: task_mapping.get(x, x))

    return df


def save_ability_matrix(df: pd.DataFrame, path: str) -> None:
    """
    Save the ability matrix to CSV.

    Args:
        df: Ability matrix DataFrame
        path: Output path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path)
    print(f"Saved ability matrix to {path}")


def load_agent_idata(
    base_path: str,
    agents: List[str],
) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    """
    Load InferenceData for multiple agents from NetCDF files.

    Also loads capability metadata if available (from *_meta.json files).

    Args:
        base_path: Base filename prefix (e.g., './data/results/agents')
        agents: List of agent names

    Returns:
        Tuple of:
        - Dictionary mapping agent names to InferenceData objects
        - List of capability column names in model parameter order (if metadata found),
          or None if no metadata available
    """
    az = _get_arviz()
    all_idata = {}
    capability_cols = None

    for agent in agents:
        filepath = f"{base_path}_{agent}.nc"
        if os.path.exists(filepath):
            print(f"Loading data for {agent} from {filepath}...")
            all_idata[agent] = az.from_netcdf(filepath)

            # Try to load capability metadata (only need to load once)
            if capability_cols is None:
                meta_path = f"{base_path}_{agent}_meta.json"
                if os.path.exists(meta_path):
                    with open(meta_path, "r") as f:
                        metadata = json.load(f)
                    capability_cols = metadata.get("capability_cols")
                    print(f"  - Loaded capability metadata: {len(capability_cols)} capabilities")
        else:
            print(f"WARNING: File not found for {agent} at {filepath}. Skipping.")

    return all_idata, capability_cols


def save_agent_idata(
    idata: Any,
    agent_name: str,
    base_path: str,
    capability_cols: Optional[List[str]] = None,
) -> str:
    """
    Save InferenceData for an agent to NetCDF, with optional capability metadata.

    Args:
        idata: ArviZ InferenceData object
        agent_name: Name of the agent
        base_path: Base path for output (without agent name)
        capability_cols: List of capability column names in model parameter order.
                         If provided, saved as JSON metadata alongside the NetCDF file.

    Returns:
        Path to saved NetCDF file
    """
    os.makedirs(os.path.dirname(base_path), exist_ok=True)
    filepath = f"{base_path}_{agent_name}.nc"
    idata.to_netcdf(filepath)
    print(f"Saved inference data to {filepath}")

    # Save capability metadata if provided
    if capability_cols is not None:
        meta_path = f"{base_path}_{agent_name}_meta.json"
        metadata = {
            "capability_cols": capability_cols,
            "agent_name": agent_name,
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved capability metadata to {meta_path}")

    return filepath


def save_population_idata(
    idata: Any,
    base_path: str,
    participant_names: List[str],
    capability_cols: List[str],
) -> str:
    """
    Save hierarchical population InferenceData to NetCDF with metadata.

    Args:
        idata: ArviZ InferenceData from build_population_model
        base_path: Output path prefix (without extension)
        participant_names: List of participant identifiers
        capability_cols: List of capability names in model parameter order

    Returns:
        Path to saved NetCDF file
    """
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    filepath = f"{base_path}_population.nc"
    idata.to_netcdf(filepath)
    print(f"Saved population inference data to {filepath}")

    meta_path = f"{base_path}_population_meta.json"
    metadata = {
        "participant_names": participant_names,
        "capability_cols": capability_cols,
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved population metadata to {meta_path}")

    return filepath


def load_population_idata(
    base_path: str,
) -> Tuple[Any, List[str], List[str]]:
    """
    Load hierarchical population InferenceData and metadata.

    Args:
        base_path: Base path prefix used when saving (without '_population.nc')

    Returns:
        Tuple of (InferenceData, participant_names, capability_cols)
    """
    az = _get_arviz()
    filepath = f"{base_path}_population.nc"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Population inference data not found at {filepath}")

    idata = az.from_netcdf(filepath)
    print(f"Loaded population inference data from {filepath}")

    meta_path = f"{base_path}_population_meta.json"
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Population metadata not found at {meta_path}")

    with open(meta_path, "r") as f:
        metadata = json.load(f)

    participant_names = metadata["participant_names"]
    capability_cols = metadata["capability_cols"]
    print(f"  {len(participant_names)} participants, {len(capability_cols)} capabilities")

    return idata, participant_names, capability_cols


def load_model_data(path: str) -> dict:
    """
    Load pickled model data.

    Args:
        path: Path to pickle file

    Returns:
        Dictionary of model objects
    """
    with open(path, "rb") as f:
        return dill.load(f)


def save_model_data(data: dict, path: str) -> None:
    """
    Save model data to pickle.

    Args:
        data: Dictionary of model objects
        path: Output path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        dill.dump(data, f)
    print(f"Saved model data to {path}")


def load_performance_data(
    results_path: str,
    annotations_path: str,
    agent_name: str = "agent",
    score_col: str = "score",
    capability_filter: Optional[List[str]] = None,
    abilities_df: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Load and merge performance results with annotations.

    Args:
        results_path: Path to results CSV with performance scores
        annotations_path: Path to annotations CSV with item demands
        agent_name: Name to use for the agent
        score_col: Column name containing performance scores
        capability_filter: Optional list of capability names or acronyms to keep
        abilities_df: DataFrame with 'Abilities' and 'Acronym' columns for acronym resolution

    Returns:
        Tuple of (performance_df, demand_matrix, item_index, capability_cols)
    """
    # Load data
    annotations_df, capability_cols, _ = load_annotations(
        annotations_path,
        capability_filter=capability_filter,
        abilities_df=abilities_df,
    )
    results_df = pd.read_csv(results_path)

    # Deduplicate results: keep the row with a valid score if one exists, else first occurrence
    n_before = len(results_df)
    results_df = results_df.sort_values(score_col, na_position="last").drop_duplicates(
        subset=["dataset_name", "sample_id"], keep="first"
    )
    n_dropped = n_before - len(results_df)
    if n_dropped > 0:
        print(f"    Note: dropped {n_dropped} duplicate result rows (kept highest score per item)")

    # Merge on identifiers
    merged = annotations_df.merge(
        results_df,
        on=["dataset_name", "sample_id"],
        how="inner",
    )

    # Create composite index
    merged["composite_id"] = (
        merged["dataset_name"].astype(str) + "_" + merged["sample_id"].astype(str)
    )

    # Extract demand matrix
    D = merged[capability_cols].to_numpy(float)
    item_index = merged["composite_id"].values

    # Create performance DataFrame
    Ym_df = pd.DataFrame(
        merged[score_col].values.reshape(1, -1),
        index=[agent_name],
        columns=item_index,
    )

    return Ym_df, D, item_index, capability_cols
