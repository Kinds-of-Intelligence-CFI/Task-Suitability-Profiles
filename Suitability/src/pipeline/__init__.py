"""Pipeline modules for the capability profiling workflow."""

from .simulation import simulate_agent_performance, create_agent_profiles

# Conditional imports for modules with heavy dependencies (arviz, pymc)
try:
    from .inference import run_inference, run_inference_batch
    from .suitability import compute_suitability_scores, score_all_tasks
except ImportError:
    pass  # Allow package to load even without full dependencies

# Questionnaire module (no heavy dependencies)
from .questionnaire import (
    load_questionnaire_data,
    apply_quality_control,
    build_ability_matrix,
)
