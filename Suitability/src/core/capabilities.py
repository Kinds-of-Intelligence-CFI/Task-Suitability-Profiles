"""
Capability name standardization and validation utilities.
"""

import pandas as pd
from typing import List, Set


# Mapping for standardizing capability names across different data sources
CAPABILITY_NAME_MAP = {
    "Inhibitory Control": "Attention and Inhibitory Control",
}


def standardize_capability_names(names: List[str]) -> List[str]:
    """
    Standardize capability column names to canonical form.

    Args:
        names: List of capability names to standardize

    Returns:
        List of standardized capability names
    """
    return [CAPABILITY_NAME_MAP.get(name, name) for name in names]


def validate_capability_alignment(
    capability_cols: List[str],
    reference_abilities_df: pd.DataFrame,
    context: str = ""
) -> None:
    """
    Validate that capability columns match the reference abilities.

    Args:
        capability_cols: List of capability column names from data
        reference_abilities_df: DataFrame with 'Abilities' column as reference
        context: String to include in error messages for debugging

    Raises:
        ValueError: If there are mismatches between columns and reference
    """
    reference_abilities = set(reference_abilities_df["Abilities"].tolist())
    data_abilities = set(capability_cols)

    missing_in_data = reference_abilities - data_abilities
    extra_in_data = data_abilities - reference_abilities

    errors = []
    if missing_in_data:
        errors.append(f"Missing abilities in data: {missing_in_data}")
    if extra_in_data:
        errors.append(f"Extra abilities in data (not in reference): {extra_in_data}")

    if errors:
        context_str = f" [{context}]" if context else ""
        raise ValueError(f"Capability alignment error{context_str}:\n" + "\n".join(errors))


def get_capability_order(abilities_df: pd.DataFrame) -> List[str]:
    """
    Get the canonical ordering of capabilities from the abilities definition file.

    Args:
        abilities_df: DataFrame with 'Abilities' column

    Returns:
        List of capability names in canonical order
    """
    return abilities_df["Abilities"].tolist()


def get_capability_acronyms(abilities_df: pd.DataFrame) -> dict:
    """
    Get mapping from full capability names to acronyms.

    Args:
        abilities_df: DataFrame with 'Abilities' and 'Acronym' columns

    Returns:
        Dictionary mapping full names to acronyms
    """
    return dict(zip(abilities_df["Abilities"], abilities_df["Acronym"]))


def resolve_capability_filter(
    filter_names: List[str],
    abilities_df: pd.DataFrame,
) -> List[str]:
    """
    Resolve a mixed list of acronyms and/or full capability names to canonical full names.

    Args:
        filter_names: List of capability identifiers (acronyms like "EM" or full names)
        abilities_df: DataFrame with 'Abilities' and 'Acronym' columns

    Returns:
        List of canonical full capability names

    Raises:
        ValueError: If any name cannot be resolved
    """
    acronym_to_full = dict(zip(abilities_df["Acronym"], abilities_df["Abilities"]))
    full_names = set(abilities_df["Abilities"])

    resolved = []
    unresolved = []
    for name in filter_names:
        if name in acronym_to_full:
            resolved.append(acronym_to_full[name])
        elif name in full_names:
            resolved.append(name)
        else:
            unresolved.append(name)

    if unresolved:
        raise ValueError(
            f"Could not resolve capability filter names: {unresolved}\n"
            f"Valid acronyms: {sorted(acronym_to_full.keys())}\n"
            f"Valid full names: {sorted(full_names)}"
        )

    return resolved
