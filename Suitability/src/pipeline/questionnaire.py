"""
Questionnaire data processing module.

Processes questionnaire data to create ability matrices that can be used
for computing suitability scores. Supports filtering by data source
(companies/online/combined) and by domain.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from numpy.linalg import norm


# Domain mapping
DOMAIN_INFO = {
    0: {"acronym": "all", "type": "All domains", "subtype": "1-6"},
    1: {"acronym": "WL", "type": "Manual-physical", "subtype": "Warehouse or logistics"},
    2: {"acronym": "MMR", "type": "Manual-physical", "subtype": "Manufacture, maintenance, or repair"},
    3: {"acronym": "NDP", "type": "Computer-digital", "subtype": "Numerical, data, or programming"},
    4: {"acronym": "AOP", "type": "Computer-office", "subtype": "Admin, organisational, or planning"},
    5: {"acronym": "CMH", "type": "Computer-office", "subtype": "Customer service, marketing or HR"},
    6: {"acronym": "HSC", "type": "Face-to-face", "subtype": "Hospitality, sales, or client care"},
}

# Default quality control thresholds
DEFAULT_COMPLETION_THRESHOLD = 100  # % completed
DEFAULT_DURATION_THRESHOLD = 600    # minimum seconds
DEFAULT_QUIZ_THRESHOLD = 50         # % correct on familiarisation quiz


def load_questionnaire_data(
    companies_path: Optional[str] = None,
    online_path: Optional[str] = None,
    tasks_path: Optional[str] = None,
    abilities_path: Optional[str] = None,
    domains_path: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load questionnaire data from CSV files.

    Args:
        companies_path: Path to company-recruited participants CSV
        online_path: Path to online-recruited participants CSV
        tasks_path: Path to tasks definition CSV
        abilities_path: Path to abilities definition CSV
        domains_path: Path to domains definition CSV

    Returns:
        Tuple of (combined_df, tasks_df, abilities_df, domains_df)
    """
    dfs = []

    if companies_path and Path(companies_path).exists():
        company_df = pd.read_csv(companies_path)
        company_df["source"] = "companies"
        dfs.append(company_df)

    if online_path and Path(online_path).exists():
        online_df = pd.read_csv(online_path)
        # Drop prescreen column if present
        if "prescreen" in online_df.columns:
            online_df = online_df.drop("prescreen", axis=1)
        online_df["source"] = "online"
        dfs.append(online_df)

    if not dfs:
        raise ValueError("At least one data source must be provided")

    df = pd.concat(dfs, ignore_index=True)

    # Load reference data
    tasks_df = pd.read_csv(tasks_path) if tasks_path else None
    abilities_df = pd.read_csv(abilities_path) if abilities_path else None
    domains_df = pd.read_csv(domains_path) if domains_path else None

    return df, tasks_df, abilities_df, domains_df


def apply_quality_control(
    df: pd.DataFrame,
    completion_threshold: float = DEFAULT_COMPLETION_THRESHOLD,
    duration_threshold: float = DEFAULT_DURATION_THRESHOLD,
    quiz_threshold: float = DEFAULT_QUIZ_THRESHOLD,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply quality control filters to questionnaire data.

    Args:
        df: Raw questionnaire DataFrame
        completion_threshold: Minimum completion percentage
        duration_threshold: Minimum duration in seconds
        quiz_threshold: Minimum quiz score percentage
        verbose: If True, print filtering statistics

    Returns:
        Filtered DataFrame
    """
    if verbose:
        print(f"Initial participants: {len(df)}")

    # Completion filter
    df = df[df["Progress"] >= completion_threshold].reset_index(drop=True)

    # Quiz score filter (Q4 familiarisation quiz)
    quiz_cols = ["Q4.12", "Q4.13", "Q4.14", "Q4.25", "Q4.26", "Q4.27"]
    existing_quiz_cols = [c for c in quiz_cols if c in df.columns]
    if existing_quiz_cols:
        df["quiz_score"] = (df[existing_quiz_cols] == 1.0).sum(axis=1) / len(existing_quiz_cols) * 100
        df = df[df["quiz_score"] >= quiz_threshold].reset_index(drop=True)

    # Duration filter
    if "Duration (in seconds)" in df.columns:
        df = df[df["Duration (in seconds)"] >= duration_threshold].reset_index(drop=True)

    if verbose:
        print(f"Participants remaining: {len(df)}")
        if "Q2.11" in df.columns:
            domain_counts = df["Q2.11"].value_counts().sort_index()
            print(f"Participants by domain: {dict(domain_counts)}")

    return df


def filter_by_source(
    df: pd.DataFrame,
    source: str = "combined",
) -> pd.DataFrame:
    """
    Filter data by source (companies, online, or combined).

    Args:
        df: Questionnaire DataFrame with 'source' column
        source: One of 'companies', 'online', or 'combined'

    Returns:
        Filtered DataFrame
    """
    if source == "combined":
        return df
    elif source in ["companies", "online"]:
        return df[df["source"] == source].reset_index(drop=True)
    else:
        raise ValueError(f"Invalid source: {source}. Must be 'companies', 'online', or 'combined'")


def filter_by_domain(
    df: pd.DataFrame,
    domain: int = 0,
) -> pd.DataFrame:
    """
    Filter data by domain.

    Args:
        df: Questionnaire DataFrame with Q2.11 (domain) column
        domain: Domain number (1-6) or 0 for all domains

    Returns:
        Filtered DataFrame
    """
    if domain == 0:
        return df
    elif domain in range(1, 7):
        return df[df["Q2.11"] == domain].reset_index(drop=True)
    else:
        raise ValueError(f"Invalid domain: {domain}. Must be 0 (all) or 1-6")


def build_ability_matrix(
    df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    abilities_df: pd.DataFrame,
    source: str = "combined",
    domain: int = 0,
    normalize: bool = False,
) -> pd.DataFrame:
    """
    Build ability matrix from questionnaire Q5.4 responses.

    The ability matrix captures the weighted importance of each cognitive
    ability for each work task, based on participant ratings.

    Args:
        df: Quality-controlled questionnaire DataFrame
        tasks_df: Tasks definition DataFrame
        abilities_df: Abilities definition DataFrame
        source: Data source filter ('companies', 'online', or 'combined')
        domain: Domain filter (0 for all, 1-6 for specific domain)
        normalize: If True, normalize rows to sum to 100

    Returns:
        DataFrame with tasks as rows, abilities as columns, weighted scores as values
    """
    # Apply filters
    df_filtered = filter_by_source(df, source)
    df_filtered = filter_by_domain(df_filtered, domain)

    if len(df_filtered) == 0:
        raise ValueError(f"No data remaining after filtering (source={source}, domain={domain})")

    # Extract Q5.4 columns (ability ratings)
    df_abilities = df_filtered.filter(like="Q5.4")

    # Convert to long format
    df_long = df_abilities.melt(var_name="task_ability", value_name="score")
    df_long = df_long.dropna(subset=["score"])

    # Parse task and ability numbers from column names
    df_long["task"] = df_long["task_ability"].str.extract(r"^(\d+)_")[0].astype(int)
    df_long["ability"] = df_long["task_ability"].str.extract(r"_(\d+)$")[0].astype(int)

    # Filter to standard tasks (1-18)
    df_long = df_long[df_long["task"] < 19]

    # Compute weighted scores
    ability_summary = (
        df_long.groupby(["task", "ability"])["score"]
        .agg(["sum", "count"])
        .reset_index()
    )
    ability_summary["weighted_score"] = ability_summary["sum"] / ability_summary["count"]

    # Create mapping dictionaries
    num2task = dict(zip(tasks_df["Number"], tasks_df["Heading"]))
    num2ability = dict(zip(abilities_df["Number"], abilities_df["Acronym"]))

    # Define ordering
    task_order = [num2task[n] for n in tasks_df["Number"]]
    ability_order = [num2ability[n] for n in abilities_df["Number"]]

    # Reshape into matrix
    ability_matrix = ability_summary.copy()
    ability_matrix["task"] = pd.Categorical(
        ability_matrix["task"].map(num2task),
        categories=task_order,
        ordered=True,
    )
    ability_matrix["ability"] = pd.Categorical(
        ability_matrix["ability"].map(num2ability),
        categories=ability_order,
        ordered=True,
    )
    ability_matrix = ability_matrix.set_index(["task", "ability"])
    ability_matrix = ability_matrix["weighted_score"].unstack(fill_value=0)

    if normalize:
        row_sums = ability_matrix.sum(axis=1)
        ability_matrix = ability_matrix.div(row_sums, axis=0) * 100

    return ability_matrix


def build_ability_matrices_by_source(
    df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    abilities_df: pd.DataFrame,
    domain: int = 0,
) -> Dict[str, pd.DataFrame]:
    """
    Build separate ability matrices for each data source.

    Args:
        df: Quality-controlled questionnaire DataFrame
        tasks_df: Tasks definition DataFrame
        abilities_df: Abilities definition DataFrame
        domain: Domain filter (0 for all, 1-6 for specific domain)

    Returns:
        Dictionary mapping source names to ability matrices
    """
    matrices = {}
    for source in df["source"].unique():
        try:
            matrices[source] = build_ability_matrix(
                df, tasks_df, abilities_df,
                source=source, domain=domain,
            )
        except ValueError:
            print(f"Warning: No data for source={source}, domain={domain}")
    return matrices


def build_ability_matrices_by_domain(
    df: pd.DataFrame,
    tasks_df: pd.DataFrame,
    abilities_df: pd.DataFrame,
    source: str = "combined",
) -> Dict[int, pd.DataFrame]:
    """
    Build separate ability matrices for each domain.

    Args:
        df: Quality-controlled questionnaire DataFrame
        tasks_df: Tasks definition DataFrame
        abilities_df: Abilities definition DataFrame
        source: Data source filter

    Returns:
        Dictionary mapping domain numbers to ability matrices
    """
    matrices = {}
    for domain in range(7):  # 0 (all) + 1-6
        try:
            matrices[domain] = build_ability_matrix(
                df, tasks_df, abilities_df,
                source=source, domain=domain,
            )
        except ValueError:
            if domain != 0:  # Only warn for specific domains
                print(f"Warning: No data for source={source}, domain={domain}")
    return matrices


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> Tuple[float, int]:
    """
    Compute cosine similarity between two arrays, handling NaN values.

    Args:
        x: First array
        y: Second array

    Returns:
        Tuple of (similarity score, number of valid comparisons)
    """
    mask = ~np.isnan(x) & ~np.isnan(y)
    n_valid = np.sum(mask)
    if n_valid == 0:
        return np.nan, 0

    x_masked, y_masked = x[mask], y[mask]
    nx, ny = norm(x_masked), norm(y_masked)

    if nx > 0 and ny > 0:
        similarity = np.dot(x_masked, y_masked) / (nx * ny)
    else:
        similarity = np.nan

    return similarity, n_valid


def compare_ability_matrices(
    matrix_a: pd.DataFrame,
    matrix_b: pd.DataFrame,
    name_a: str = "A",
    name_b: str = "B",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare two ability matrices using correlation and cosine similarity.

    Args:
        matrix_a: First ability matrix
        matrix_b: Second ability matrix
        name_a: Name for first matrix
        name_b: Name for second matrix

    Returns:
        Tuple of (column-wise comparison, row-wise comparison) DataFrames
    """
    # Align matrices
    common_tasks = matrix_a.index.intersection(matrix_b.index)
    common_abilities = matrix_a.columns.intersection(matrix_b.columns)

    A = matrix_a.loc[common_tasks, common_abilities]
    B = matrix_b.loc[common_tasks, common_abilities]

    # Column-wise (abilities) comparison
    colwise_pearson = A.corrwith(B)
    colwise_results = [
        cosine_similarity(A.iloc[:, i].to_numpy(), B.iloc[:, i].to_numpy())
        for i in range(A.shape[1])
    ]
    colwise_cosine, colwise_n = zip(*colwise_results) if colwise_results else ([], [])

    colwise_df = pd.DataFrame({
        "ability": A.columns,
        f"pearson_{name_a}_vs_{name_b}": colwise_pearson.values,
        f"cosine_{name_a}_vs_{name_b}": colwise_cosine,
        "n_valid_comparisons": colwise_n,
    }).set_index("ability")

    # Row-wise (tasks) comparison
    rowwise_pearson = A.T.corrwith(B.T)
    rowwise_results = [
        cosine_similarity(A.iloc[i, :].to_numpy(), B.iloc[i, :].to_numpy())
        for i in range(A.shape[0])
    ]
    rowwise_cosine, rowwise_n = zip(*rowwise_results) if rowwise_results else ([], [])

    rowwise_df = pd.DataFrame({
        "task": A.index,
        f"pearson_{name_a}_vs_{name_b}": rowwise_pearson.values,
        f"cosine_{name_a}_vs_{name_b}": rowwise_cosine,
        "n_valid_comparisons": rowwise_n,
    }).set_index("task")

    return colwise_df, rowwise_df


def get_domain_label(domain: int) -> str:
    """Get human-readable label for a domain number."""
    info = DOMAIN_INFO.get(domain, {})
    if domain == 0:
        return "All domains"
    return f"{info.get('acronym', '')} ({info.get('subtype', '')})"


def get_participant_stats(
    df: pd.DataFrame,
    source: str = "combined",
    domain: int = 0,
) -> Dict[str, int]:
    """
    Get participant statistics after filtering.

    Args:
        df: Quality-controlled questionnaire DataFrame
        source: Data source filter
        domain: Domain filter

    Returns:
        Dictionary with participant counts
    """
    df_filtered = filter_by_source(df, source)
    df_filtered = filter_by_domain(df_filtered, domain)

    stats = {
        "total": len(df_filtered),
        "by_source": df_filtered["source"].value_counts().to_dict() if "source" in df_filtered else {},
    }

    if "Q2.11" in df_filtered.columns:
        stats["by_domain"] = df_filtered["Q2.11"].value_counts().to_dict()

    return stats
