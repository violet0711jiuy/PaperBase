"""Requirement shape authority and explicit legacy/direct-caller fallback."""
from .contracts import EvidenceProjectionType, Intent, ResearchGoal


def _legacy_projection_type(context):
    if context.retrieval_goal is ResearchGoal.EXHAUSTIVE_ENUMERATION:
        return EvidenceProjectionType.ENUMERATION
    if context.intent is Intent.COMPARE or context.retrieval_goal is ResearchGoal.COMPARISON_SUPPORT:
        return EvidenceProjectionType.COMPARISON
    question = context.resolved_query.casefold()
    if any(marker in question for marker in ("如何", "怎么", "机制", "how", "mechanism")):
        return EvidenceProjectionType.MECHANISM
    return EvidenceProjectionType.FACT


def requirement_projection_type(requirement, *, context):
    if requirement.projection_type is not None:
        return requirement.projection_type
    return _legacy_projection_type(context)
