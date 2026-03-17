"""CoQA (Conversational Question Answering) evaluation task."""

from .CoQA_task import record_to_sample, coqa_task

__all__ = ["record_to_sample", "coqa_task"]