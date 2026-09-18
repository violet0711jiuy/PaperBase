"""PaperBase 单 Agent 研究编排与证据治理层。"""

from .contracts import (
    AgentAction,
    CorpusScopeSnapshot,
    CoverageState,
    CoverageStrategy,
    EvidenceOwnership,
    EvidenceQualification,
    EvidenceRelation,
    EvidenceRecord,
    EvidenceTarget,
    FinishAction,
    GetPaperOverviewAction,
    GetSectionContextAction,
    Intent,
    QueryContext,
    QueryScope,
    ResearchGoal,
    ResearchPlan,
    ResearchRunResult,
    ResearchRunStatus,
    ResearchSubtask,
    PaperAssessment,
    PaperAssessmentStatus,
    ScopeResolution,
    ScopeResolutionStatus,
    ScopeSource,
    SearchKnowledgeBaseAction,
    SearchPapersAction,
    SearchReferencesAction,
    StopReason,
)
from .decision import LLMDecisionAgent, ScriptedDecisionAgent
from .coverage import CoverageManager, CoveragePolicy
from .evidence_registry import EvidenceRegistry
from .factory import AgenticResearchConfigurationError, create_agentic_research_service
from .graph import (
    AgentGraphContext,
    AgentGraphState,
    GraphExecution,
    LangGraphResearchWorkflow,
    build_agentic_research_graph,
)
from .planner import ResearchPlanner
from .query_understanding import QueryUnderstandingService
from .qualification import EvidenceQualificationManager
from .runtime import ResearchRuntime
from .scope_resolver import ScopeResolver, TrustedWorkspaceScope
from .selector import FinalEvidenceSelector
from .service import AgenticResearchService
from .tools import ResearchToolExecutor

__all__ = [
    "AgentAction",
    "CorpusScopeSnapshot",
    "CoverageManager",
    "CoveragePolicy",
    "CoverageState",
    "CoverageStrategy",
    "EvidenceOwnership",
    "EvidenceQualification",
    "EvidenceQualificationManager",
    "EvidenceRelation",
    "EvidenceRecord",
    "EvidenceTarget",
    "FinishAction",
    "FinalEvidenceSelector",
    "GetPaperOverviewAction",
    "GetSectionContextAction",
    "Intent",
    "QueryContext",
    "QueryScope",
    "ResearchGoal",
    "ResearchPlan",
    "ResearchRunResult",
    "ResearchRunStatus",
    "ResearchSubtask",
    "PaperAssessment",
    "PaperAssessmentStatus",
    "ScopeResolution",
    "ScopeResolutionStatus",
    "ScopeSource",
    "SearchKnowledgeBaseAction",
    "SearchPapersAction",
    "SearchReferencesAction",
    "StopReason",
    "AgenticResearchService",
    "AgenticResearchConfigurationError",
    "AgentGraphContext",
    "AgentGraphState",
    "EvidenceRegistry",
    "GraphExecution",
    "LangGraphResearchWorkflow",
    "LLMDecisionAgent",
    "QueryUnderstandingService",
    "ResearchPlanner",
    "ResearchRuntime",
    "ResearchToolExecutor",
    "ScopeResolver",
    "ScriptedDecisionAgent",
    "TrustedWorkspaceScope",
    "build_agentic_research_graph",
    "create_agentic_research_service",
]
