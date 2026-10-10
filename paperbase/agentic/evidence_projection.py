"""Evidence-local, Requirement-directed multi-projection of registered Evidence.

Legacy analyzer callers use the explicit adapter in evidence_analysis;
checkpoint compatibility parses stored assessments without rerunning extraction.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from typing import Iterable

from pydantic import ValidationError

from paperbase.llm.client import ChatCompletionClient, LLMCompletion, ValidationDiagnostic, complete_json_with_diagnostics, structured_client_view
from paperbase.llm.diagnostics import classify_llm_error, failure_record, repair_hint, success_record
from paperbase.prompts.evidence_projector import (
    EVIDENCE_PROJECTOR_BATCH_INSTRUCTION,
    EVIDENCE_PROJECTOR_SYSTEM_PROMPT,
    build_evidence_projector_user_prompt,
)

from .contracts import (
    ComparisonProjection,
    EnumerationProjection,
    EvidenceAnalysis,
    EvidenceAnalysisCacheStats,
    EvidenceAnalysisStatus,
    EvidenceOwnership,
    EvidenceProjectionType,
    EvidenceRecord,
    FactItem,
    FactProjection,
    Intent,
    MechanismProjection,
    ProjectedEntity,
    ProjectedEntityRole,
    QueryContext,
    ResearchGoal,
    ResearchPlan,
    RequirementSupportAnchor,
)
from .evidence_analysis import EvidenceAnalysisRequest, EvidenceAnalysisResult, requires_analysis, merge_analysis_groups
from .evidence_analysis_cache import SQLiteEvidenceAnalysisCache, EVIDENCE_ANALYSIS_SCHEMA_VERSION
from .projector_trace import pre_parse_summary, post_conversion_summary
from .projection_shapes import requirement_projection_type, _legacy_projection_type
from .projection_semantics import (
    build_projection_semantic_request, projection_semantic_cache_context,
    scoped_projection_requirements,
)
from .grounding import (
    entity_attribution_valid, local_source_window, source_contains,
    support_rejection, verified_support_ids,
)
from .source_spans import (
    SourceSpanResolutionError,
    resolve_source_span_ids,
    source_spans_payload,
)
from .schemas import (
    transport_projection_model,
    ComparisonProjectionDraft,
    ComparisonProjectionBatchDraft,
    EnumerationProjectionDraft,
    EnumerationProjectionBatchDraft,
    EvidenceProjectionBatchDraft,
    FactProjectionDraft,
    FactProjectionBatchDraft,
    MechanismProjectionDraft,
    MechanismProjectionBatchDraft,
)


@dataclass(frozen=True)
class ProjectionRequest:
    question: str
    context: QueryContext
    plan: ResearchPlan
    projection_requirement_ids: tuple[str, ...]
    projection_type: EvidenceProjectionType | None = None


@dataclass(frozen=True)
class BatchProjectorInput:
    request: ProjectionRequest
    projection_type: EvidenceProjectionType
    evidences: tuple[EvidenceRecord, ...]

    def to_payload(self) -> dict[str, object]:
        payload = _prompt_payload(self.request, self.projection_type, self.evidences)
        payload["goal"] = self.request.plan.goal
        payload["evidence"] = [dict(_payload(record, self.request.plan), evidence_id=_transport_evidence_id(record),
            projection_type=self.projection_type.value) for record in self.evidences]
        return payload


def _transport_evidence_id(record: EvidenceRecord) -> str:
    """Keep source identities distinct from Requirement IDs on the batch wire.

    The source fingerprint is stable across packing/order/runtime ID changes and never enters the
    registry, cache key or production Projection contract.
    """
    return "source:" + record.fingerprint


class LLMEvidenceProjector:
    """Project only the task-shaped data needed by the projection Requirement scope."""

    def __init__(
        self,
        client: ChatCompletionClient,
        *,
        max_attempts: int = 2,
        cache: SQLiteEvidenceAnalysisCache | None = None,
        batch_size: int = 4,
        batch_base_tokens: int = 200,
        batch_per_evidence_tokens: int = 1400,
        batch_max_output_tokens: int = 8000,
        single_max_output_tokens: int = 1400,
        temperature: float = 0,
    ) -> None:
        self._client = structured_client_view(client, temperature=temperature)
        if not 1 <= batch_size <= 32 or max_attempts < 1:
            raise ValueError("Invalid projector batch size or attempts.")
        if batch_base_tokens < 0 or min(batch_per_evidence_tokens, batch_max_output_tokens, single_max_output_tokens) < 1:
            raise ValueError("Projector output budgets must be positive.")
        self._max_attempts = min(max_attempts, 2)
        self._cache = cache
        self._batch_size = batch_size
        # Legacy token-budget arguments are accepted but intentionally unused.

    def analyze(self, *, request: EvidenceAnalysisRequest) -> EvidenceAnalysisResult:
        pending = tuple(record for record in request.evidence if requires_analysis(record, request.plan))
        requirements = {r.requirement_id: r for r in request.plan.requirements}
        groups = {}
        partials = {record.evidence_id: [] for record in pending}
        for record in pending:
            missing = tuple(rid for rid in request.scope_for(record.evidence_id)
                            if rid not in record.assessment.evaluated_requirement_ids)
            by_type = {}
            for rid in missing:
                shape = requirement_projection_type(requirements[rid], context=request.context)
                by_type.setdefault(shape, []).append(rid)
            for shape, ids in by_type.items():
                # Partial calls do not reproduce already evaluated sibling shapes.
                groups.setdefault((shape, tuple(ids)), []).append(replace(record, assessment=None))
        calls, diagnostics, stats = 0, [], EvidenceAnalysisCacheStats()
        for (shape, scope), records in groups.items():
            group_request = ProjectionRequest(request.question, request.context, request.plan, scope, shape)
            result = self._analyze_scope(request=group_request, evidence=tuple(records))
            for item in result.analyses:
                partials[item.evidence_id].append(item)
            calls += result.llm_calls
            diagnostics.extend(result.llm_diagnostics)
            stats += result.cache_stats
        analyses = tuple(merge_analysis_groups(record, tuple(partials[record.evidence_id]), request.plan)
                         for record in pending)
        return EvidenceAnalysisResult(analyses, calls, tuple(diagnostics), stats)

    def _analyze_scope(self, *, request: ProjectionRequest,
                       evidence: tuple[EvidenceRecord, ...]) -> EvidenceAnalysisResult:
        question = request.question
        projection_type = _request_projection_type(request)
        semantic_request = build_projection_semantic_request(
            question=question, plan=request.plan, projection_type=projection_type,
            requirement_ids=request.projection_requirement_ids,
        )
        cache_context = projection_semantic_cache_context(semantic_request)
        pending = evidence
        # Registry already deduplicates fingerprints. Also tolerate direct callers
        # providing the same source under different runtime IDs.
        representatives: dict[str, EvidenceRecord] = {}
        aliases: dict[str, str] = {}
        unique_pending = []
        for record in pending:
            original = representatives.setdefault(record.fingerprint, record)
            if original.evidence_id == record.evidence_id:
                unique_pending.append(record)
            else:
                aliases[record.evidence_id] = original.evidence_id
        by_id: dict[str, EvidenceAnalysis] = {}
        cache_stats = EvidenceAnalysisCacheStats()
        uncached: list[EvidenceRecord] = []
        for record in unique_pending:
            if self._cache is None:
                uncached.append(record)
                continue
            lookup = self._cache.lookup(
                question=question, record=record, projection_context=cache_context
            )
            if lookup.analysis is not None and (
                bool(lookup.analysis.projections)
                or lookup.analysis.status is EvidenceAnalysisStatus.IRRELEVANT
            ):
                # A legacy cache entry cannot attest an evaluation scope.
                candidate = _revalidate_cached(record, lookup.analysis, request.plan)
                try:
                    if set(candidate.evaluated_requirement_ids) != set(request.projection_requirement_ids):
                        raise ValueError("Cache lacks the complete evaluated scope.")
                except ValueError:
                    uncached.append(record)
                    cache_stats += EvidenceAnalysisCacheStats(misses=1)
                else:
                    by_id[record.evidence_id] = candidate
                    cache_stats += EvidenceAnalysisCacheStats(hits=1)
            else:
                uncached.append(record)
                cache_stats = cache_stats + EvidenceAnalysisCacheStats(misses=1, errors=int(lookup.error))

        diagnostics = []
        calls = 0
        for offset in range(0, len(uncached), self._batch_size):
            batch = tuple(uncached[offset:offset + self._batch_size])
            analyses, batch_calls, batch_diagnostics = self._project_batch(request, batch)
            calls += batch_calls
            diagnostics.extend(batch_diagnostics)
            for analysis in analyses:
                by_id[analysis.evidence_id] = analysis
                if self._cache is not None and analysis.status in {
                    EvidenceAnalysisStatus.QUALIFIED, EvidenceAnalysisStatus.IRRELEVANT
                }:
                    record = next(item for item in batch if item.evidence_id == analysis.evidence_id)
                    wrote = self._cache.put(
                        question=question,
                        record=record,
                        analysis=analysis,
                        projection_context=cache_context,
                    )
                    cache_stats = cache_stats + EvidenceAnalysisCacheStats(writes=int(wrote), errors=int(not wrote))
        for record in pending:
            if record.evidence_id in aliases:
                original = by_id[aliases[record.evidence_id]]
                candidate = replace(
                    original, evidence_id=record.evidence_id,
                    projection=None,
                    projections=tuple(replace(part, evidence_id=record.evidence_id) for part in original.projections),
                )
                by_id[record.evidence_id] = candidate
        return EvidenceAnalysisResult(
            analyses=tuple(by_id[item.evidence_id] for item in pending),
            llm_calls=calls,
            llm_diagnostics=tuple(diagnostics),
            cache_stats=cache_stats,
        )

    def _project_batch(
        self, request: ProjectionRequest, batch: tuple[EvidenceRecord, ...]
    ) -> tuple[tuple[EvidenceAnalysis, ...], int, tuple]:
        projection_type = _request_projection_type(request)
        if len(batch) == 1:
            return self._project_with_recovery(request, batch, projection_type)
        return self._project_transport_batch(request, batch, projection_type)

    def output_budget(self, size: int) -> None:
        """Structured output is limited by the provider, not a local cap."""
        return None

    def _project_transport_batch(self, request, batch, projection_type):
        diagnostics = []
        retained: dict[str, EvidenceAnalysis] = {}
        feedback = []
        repair_context = None
        reason = "batch_validation_failed"
        budget = self.output_budget(len(batch))
        # Exactly one repair at most, regardless of the single-call retry setting.
        for attempt in range(1, min(self._max_attempts, 2) + 1):
            completion = None
            pre_summary = post_summary = None
            try:
                payload = BatchProjectorInput(request, projection_type, batch).to_payload()
                if feedback:
                    payload["output_validation_feedback"] = feedback
                if repair_context is not None:
                    payload["repair_context"] = repair_context
                schema = transport_projection_model(projection_type.value).model_json_schema()
                completion = complete_json_with_diagnostics(
                    self._client,
                    system_prompt=EVIDENCE_PROJECTOR_SYSTEM_PROMPT + "\n" + EVIDENCE_PROJECTOR_BATCH_INSTRUCTION,
                    user_prompt=build_evidence_projector_user_prompt(payload),
                    json_schema=schema,
                    schema_name="paperbase_evidence_transport_batch",
                )
                pre_summary = pre_parse_summary(content=completion.content, request=request, records=batch,
                    projection_type=projection_type, schema_name="paperbase_evidence_transport_batch",
                    schema=schema, schema_version=EVIDENCE_ANALYSIS_SCHEMA_VERSION,
                    call_mode="batch", attempt=attempt)
                valid, violations = _validate_transport_items(
                    batch, request, projection_type, completion.content
                )
                post_summary = post_conversion_summary(analyses=valid, records=batch, request=request,
                    conversion_succeeded=not violations)
                # Once an item passes the existing production schema/integrity,
                # a later malformed batch must not erase it.
                for analysis in valid:
                    retained.setdefault(analysis.evidence_id, analysis)
                if completion.finish_reason == "length":
                    raise ValueError("Provider truncated structured output.")
                if violations:
                    raise BatchStructureError(violations)
                diagnostic = success_record(node="project_evidence", attempt=attempt, completion=completion)
            except Exception as error:
                stage = classify_llm_error(error)
                reason = stage
                if isinstance(error, BatchStructureError):
                    reason = "batch_structure:" + "|".join(dict.fromkeys(
                        str(item["code"]) for item in error.violations
                    ))[:200]
                recoverable = stage in {"empty_content", "invalid_json", "schema_validation", "domain_validation"}
                truncated = _is_truncated(completion, error)
                can_retry = recoverable and not truncated and attempt < min(self._max_attempts, 2)
                if truncated:
                    reason = "truncation"
                diagnostic = failure_record(node="project_evidence", attempt=attempt,
                    error=error, completion=completion, can_retry=can_retry,
                    validation_diagnostics=_projection_validation_diagnostics(error, attempt, can_retry, batch))
                if truncated:
                    diagnostic = replace(diagnostic, error_stage="truncation", recovery_action="fallback_single")
                if isinstance(error, BatchStructureError):
                    diagnostic = replace(diagnostic, validation_summary=";".join(
                        f"{item['code']}@{item['field_path']}:{item['evidence_id']}"
                        for item in error.violations[:3]
                    )[:300])
                feedback = [_projection_repair_feedback(error)]
                repair_context = _repair_context_from_completion(completion) if can_retry else None
                diagnostics.append(replace(diagnostic, projector_input_size=len(batch),
                    requested_output_tokens=budget, projector_call_mode="batch",
                    projector_repair=attempt > 1,
                    projector_pre_parse_summary=pre_summary, projector_post_conversion_summary=post_summary))
                if can_retry:
                    continue
                break
            diagnostics.append(replace(diagnostic, projector_input_size=len(batch),
                requested_output_tokens=budget, projector_call_mode="batch",
                projector_repair=attempt > 1,
                projector_pre_parse_summary=pre_summary, projector_post_conversion_summary=post_summary))
            return tuple(retained[item.evidence_id] for item in batch), len(diagnostics), tuple(diagnostics)

        diagnostics[-1] = replace(diagnostics[-1], projector_fallback_reason=reason)
        for record in batch:
            if record.evidence_id in retained:
                continue
            analyses, _, single_diagnostics = self._project_with_recovery(
                request, (record,), projection_type, fallback_reason=reason
            )
            retained[record.evidence_id] = analyses[0]
            diagnostics.extend(single_diagnostics)
        return tuple(retained[item.evidence_id] for item in batch), len(diagnostics), tuple(diagnostics)

    def _project_with_recovery(
        self,
        request: ProjectionRequest,
        batch: tuple[EvidenceRecord, ...],
        projection_type: EvidenceProjectionType,
        *, fallback_reason: str | None = None,
    ) -> tuple[tuple[EvidenceAnalysis, ...], int, tuple]:
        """Retry recoverable output errors without removing Evidence text."""
        diagnostics = []
        validation_feedback: list[dict[str, object]] = []
        repair_context = None
        for attempt in range(1, self._max_attempts + 1):
            completion: LLMCompletion | None = None
            pre_summary = post_summary = None
            try:
                payload = _prompt_payload(request, projection_type, batch)
                if validation_feedback:
                    payload["output_validation_feedback"] = validation_feedback
                if repair_context is not None:
                    payload["repair_context"] = repair_context
                schema = _projection_schema(projection_type)
                completion = complete_json_with_diagnostics(
                    self._client,
                    system_prompt=EVIDENCE_PROJECTOR_SYSTEM_PROMPT,
                    user_prompt=build_evidence_projector_user_prompt(
                        payload
                    ),
                    json_schema=schema,
                    schema_name="paperbase_evidence_projection",
                )
                pre_summary = pre_parse_summary(content=completion.content, request=request, records=batch,
                    projection_type=projection_type, schema_name="paperbase_evidence_projection",
                    schema=schema, schema_version=EVIDENCE_ANALYSIS_SCHEMA_VERSION,
                    call_mode="fallback_single" if fallback_reason else "single",
                    attempt=attempt, fallback_reason=fallback_reason)
                draft = _parse_projection_batch(projection_type, completion.content)
                if completion.finish_reason == "length":
                    raise ValueError("Provider truncated structured output.")
                analyses = _convert(batch, request, projection_type, draft)
                post_summary = post_conversion_summary(analyses=analyses, records=batch, request=request,
                    conversion_succeeded=True)
                diagnostics.append(replace(success_record(
                    node="project_evidence", attempt=attempt, completion=completion
                ), projector_input_size=1, requested_output_tokens=None,
                    projector_call_mode="fallback_single" if fallback_reason else "single",
                    projector_repair=attempt > 1, projector_fallback_reason=fallback_reason,
                    projector_pre_parse_summary=pre_summary, projector_post_conversion_summary=post_summary))
                return analyses, attempt, tuple(diagnostics)
            except Exception as error:
                stage = classify_llm_error(error)
                recoverable = stage in {
                    "empty_content", "invalid_json", "schema_validation", "domain_validation"
                }
                truncated = _is_truncated(completion, error)
                can_retry = recoverable and not truncated and attempt < self._max_attempts
                diagnostics.append(replace(failure_record(
                    node="project_evidence", attempt=attempt, error=error,
                    completion=completion, can_retry=can_retry,
                    validation_diagnostics=_projection_validation_diagnostics(error, attempt, can_retry, batch),
                ), projector_input_size=1, requested_output_tokens=None,
                    **({"error_stage": "truncation", "recovery_action": "failed_evidence"} if truncated else {}),
                    projector_call_mode="fallback_single" if fallback_reason else "single",
                    projector_repair=attempt > 1, projector_fallback_reason=fallback_reason,
                    projector_pre_parse_summary=pre_summary, projector_post_conversion_summary=post_summary))
                if not can_retry:
                    return _failed(batch), attempt, tuple(diagnostics)
                validation_feedback.append(_projection_repair_feedback(error, evidence_id=batch[0].evidence_id))
                repair_context = _repair_context_from_completion(completion)
        return _failed(batch), self._max_attempts, tuple(diagnostics)


class BatchStructureError(ValueError):
    def __init__(self, violations: list[dict[str, object]]) -> None:
        super().__init__("Evidence batch failed structural validation.")
        self.violations = violations


class ProjectionIntegrityError(ValueError):
    """Existing support rejection with safe, specific repair diagnostics.

    Does not change which assertions are accepted. Raw excerpts and arbitrary
    model-supplied IDs must never enter these diagnostics.
    """

    def __init__(self, evidence_id, allowed, rejections):
        super().__init__("Projection group integrity failed: asserted support has invalid provenance.")
        constraints = {
            "missing_source_anchor": ("support_candidates", "Each asserted Requirement needs a source candidate."),
            "source_span_missing": ("support_candidates.source_span_ids", "Select at least one source span ID from this Evidence."),
            "source_span_not_found": ("support_candidates.source_span_ids", "Use only source span IDs listed on this Evidence."),
            "source_spans_duplicate": ("support_candidates.source_span_ids", "Do not repeat a source span ID."),
            "source_spans_out_of_order": ("support_candidates.source_span_ids", "List source span IDs in source order."),
            "source_spans_non_contiguous": ("support_candidates.source_span_ids", "Multiple source span IDs must be consecutive within this Evidence."),
            "source_spans_too_broad": ("support_candidates.source_span_ids", "Select IDs whose smallest enclosing source envelope is local and bounded."),
            "fact_ownership_unknown": ("ownership", "Support requires grounded fact ownership; do not guess ownership."),
            "requirement_id_mismatch": ("support_candidates.requirement_id", "The candidate must target its asserted Requirement."),
            "not_proposed": ("supported_requirement_ids", "Each candidate Requirement must also be asserted in supported_requirement_ids."),
            "no_attributed_entity": ("entities", "Enumeration support requires an entity with valid source attribution."),
        }
        self.violations = []
        for rejection in dict.fromkeys(rejections):
            if not rejection.startswith("support_rejected:"):
                continue
            _, requirement_id, code = rejection.split(":", 2)
            path, constraint = constraints.get(code, (
                "support_candidates", "Asserted support must satisfy the existing provenance contract."
            ))
            self.violations.append({
                "evidence_id": evidence_id,
                "requirement_id": requirement_id if requirement_id in allowed else None,
                "field_path": "projections." + path,
                "code": code if code in constraints else "support_provenance_invalid",
                "validator_name": "ProjectorIntegrity",
                "expected_constraint": constraint,
            })


def _validate_transport_items(records, request, projection_type, content):
    """Parse the envelope first, then reuse single-Evidence production validation.

    The provider sees a strict full schema. Locally we deliberately don't call
    its whole-batch Pydantic adapter: one bad item must not hide its valid peers.
    """
    payload = json.loads(content)
    if not isinstance(payload, dict) or set(payload) != {"items"} or not isinstance(payload.get("items"), list):
        raise BatchStructureError([{
            "evidence_id": None, "field_path": "items", "code": "invalid_envelope",
            "expected_constraint": "An object containing only an items array.",
        }])
    raw_items = payload["items"]
    expected = {_transport_evidence_id(record): record for record in records}
    grouped = {evidence_id: [] for evidence_id in expected}
    violations = []

    def violation(evidence_id, path, code, constraint):
        violations.append({"evidence_id": evidence_id, "field_path": path,
                           "code": code, "expected_constraint": constraint})

    for index, item in enumerate(raw_items):
        path = f"items.{index}"
        if not isinstance(item, dict) or not isinstance(item.get("evidence_id"), str):
            violation(None, path + ".evidence_id", "invalid_id", "A string ID from the input batch.")
            continue
        evidence_id = item["evidence_id"]
        if evidence_id not in expected:
            # Do not echo arbitrary model-supplied IDs in persisted feedback.
            violation(None, path + ".evidence_id", "unknown_id", "Only input evidence IDs are allowed.")
            continue
        grouped[evidence_id].append((index, item))
    valid = []
    for evidence_id, record in expected.items():
        matches = grouped[evidence_id]
        if len(matches) != 1:
            violation(evidence_id, "items.evidence_id", "missing_id" if not matches else "duplicate_id",
                      "Exactly one batch item per input evidence_id.")
            continue
        index, item = matches[0]
        path = f"items.{index}"
        projections = item.get("projections")
        if set(item) != {"evidence_id", "projections"} or not isinstance(projections, list):
            violation(evidence_id, path, "invalid_item", "Only evidence_id and a projections array.")
            continue
        if any(not isinstance(part, dict) or part.get("evidence_id", evidence_id) != evidence_id for part in projections):
            violation(evidence_id, path + ".projections.evidence_id", "projection_id_mismatch",
                      "Every projection ID must equal its enclosing item ID.")
            continue
        try:
            if not projections:
                candidate = EvidenceAnalysis(record.evidence_id, EvidenceAnalysisStatus.IRRELEVANT,
                    evaluated_requirement_ids=request.projection_requirement_ids)
                valid.append(candidate)
                continue
            # The transport envelope is the sole identity authority. Rebind the
            # ID before using the unchanged single-Evidence production schema.
            bound = [dict(part, evidence_id=record.evidence_id) for part in projections]
            draft = _parse_projection_batch(projection_type, json.dumps({"projections": bound}))
            valid.extend(_convert((record,), request, projection_type, draft))
        except ValidationError as error:
            for detail in error.errors(include_input=False, include_url=False):
                violation(evidence_id, path + "." + ".".join(map(str, detail["loc"])),
                          detail["type"], detail["msg"])
        except (ValueError, TypeError) as error:
            # Reuse single-Evidence feedback, preserving the exact rejected
            # constraint instead of flattening provenance into a status error.
            for detail in _projection_repair_feedback(error, evidence_id=record.evidence_id).get("violations", []):
                violations.append(dict(detail, evidence_id=evidence_id,
                    runtime_evidence_id=record.evidence_id,
                    field_path=path + "." + str(detail["field_path"])))
    return tuple(valid), violations


def _projection_repair_feedback(error: Exception, *, evidence_id: str | None = None) -> dict[str, object]:
    """Return specific schema/domain constraints, never raw model/source content."""
    feedback: dict[str, object] = {
        "stage": classify_llm_error(error),
        "instruction": repair_hint(error),
    }
    if isinstance(error, (BatchStructureError, ProjectionIntegrityError)):
        feedback["violations"] = error.violations
    elif isinstance(error, ValidationError):
        feedback["violations"] = [
            {
                "field_path": ".".join(str(part) for part in item["loc"]) or "root",
                "evidence_id": evidence_id,
                "code": item["type"],
                "expected_constraint": item["msg"],
            }
            for item in error.errors(include_input=False, include_url=False)[:8]
        ]
    elif isinstance(error, (ValueError, TypeError)):
        # These constraints are existing converter branches, not new support rules.
        constraints = {
            "qualified MechanismProjection requires mechanism.": ("projections.0.mechanism", "status_branch_violation", "A qualified mechanism requires a non-null mechanism string."),
            "Evidence Projector 必须完整返回输入 evidence_id。": ("projections.evidence_id", "id_mapping_error", "Return exactly the input Evidence IDs, not Requirement IDs."),
            "Projector 返回的 projection_type 与当前任务不一致。": ("projections.projection_type", "invalid_enum", "Use the requested projection_type."),
            "supported_requirement_ids 必须属于当前 projection Requirement scope。": ("projections.supported_requirement_ids", "requirement_id_mapping_error", "Use only IDs in the projection Requirement scope."),
            "同一 Evidence 仅 Fact 类型可返回多条投影。": ("projections", "projection_cardinality", "Non-Fact Evidence requires one projection."),
            "Projection group integrity failed: asserted support has invalid provenance.": ("projections.support_candidates", "support_provenance_invalid", "Asserted support failed provenance validation; the original exception does not identify the specific rule."),
            "irrelevant Projection 不得支持 Requirement。": ("projections.supported_requirement_ids", "irrelevant_has_support", "An irrelevant projection must not assert Requirement support."),
            "irrelevant Projection 不得提出支持锚点。": ("projections.support_candidates", "irrelevant_has_candidates", "An irrelevant projection must not contain support candidates."),
            "irrelevant Projection 必须为空。": ("projections", "irrelevant_has_content", "An irrelevant projection must have empty content fields."),
            "qualified FactProjection requires fact_type, value and ownership.": ("projections", "qualified_payload_missing", "A qualified fact requires fact_type, value and ownership."),
        }
        path, code, constraint = constraints.get(str(error), (
            "projections", "projection_schema_validation" if isinstance(error, TypeError) else "projection_domain_validation",
            "The output failed conversion; no specific failed rule is available."
        ))
        feedback["violations"] = [{"evidence_id": evidence_id, "field_path": path,
                                    "code": code, "expected_constraint": constraint}]
    return feedback


def _repair_context_from_completion(completion: LLMCompletion | None) -> dict[str, object] | None:
    """Return an ephemeral previous-draft context for minimal repair only.

    The previous provider draft is never checkpointed and is not an authority.
    It is fed back only so a repair call can correct the rejected fields instead
    of silently replacing an otherwise useful semantic judgment.
    """
    if completion is None or not isinstance(completion.content, str):
        return None
    try:
        previous = json.loads(completion.content)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(previous, (dict, list)):
        return None
    return {
        "mode": "minimal_correction",
        "previous_rejected_draft": previous,
        "instruction": (
            "Repair the reported validation failures with the smallest necessary semantic change. "
            "The previous draft is not authoritative, but do not discard a source-grounded support "
            "judgment merely to avoid repairing an invalid anchor."
        ),
    }


def _is_truncated(completion, error):
    return (getattr(completion, "finish_reason", None) or getattr(error, "finish_reason", None)) == "length"


def _projection_validation_diagnostics(error, attempt, can_retry, batch):
    feedback = _projection_repair_feedback(error, evidence_id=batch[0].evidence_id if len(batch) == 1 else None)
    return tuple(ValidationDiagnostic(
        node="project_evidence", attempt=attempt, error_stage=classify_llm_error(error),
        error_code=str(v["code"]), field_path=str(v["field_path"]),
        validator_name=str(v.get("validator_name", "ProjectorStructure")),
        safe_reason="Projector output violates the reported constraint.",
        expected_constraint=str(v["expected_constraint"]),
        actual_summary=json.dumps({"code": v["code"],
            "evidence_id": v.get("runtime_evidence_id", v.get("evidence_id")),
            "requirement_id": v.get("requirement_id")}, separators=(",", ":")),
        requirement_id=v.get("requirement_id"),
        recovery_action="repair_domain" if can_retry and classify_llm_error(error) == "domain_validation"
                        else "repair_structure" if can_retry else "fallback",
    ) for v in feedback.get("violations", [])[:8])


def _request_projection_type(request: ProjectionRequest) -> EvidenceProjectionType:
    if request.projection_type is not None:
        return request.projection_type
    # Legacy direct ProjectionRequest fixtures only (public analyze sets a shape).
    return _legacy_projection_type(request.context)


def _projection_schema(projection_type: EvidenceProjectionType) -> dict[str, object]:
    return {
        EvidenceProjectionType.ENUMERATION: EnumerationProjectionBatchDraft,
        EvidenceProjectionType.COMPARISON: ComparisonProjectionBatchDraft,
        EvidenceProjectionType.FACT: FactProjectionBatchDraft,
        EvidenceProjectionType.MECHANISM: MechanismProjectionBatchDraft,
    }[projection_type].model_json_schema()


def _parse_projection_batch(
    projection_type: EvidenceProjectionType, content: str
) -> EvidenceProjectionBatchDraft | object:
    return {
        EvidenceProjectionType.ENUMERATION: EnumerationProjectionBatchDraft,
        EvidenceProjectionType.COMPARISON: ComparisonProjectionBatchDraft,
        EvidenceProjectionType.FACT: FactProjectionBatchDraft,
        EvidenceProjectionType.MECHANISM: MechanismProjectionBatchDraft,
    }[projection_type].model_validate_json(content)


def _prompt_payload(
    request: ProjectionRequest,
    projection_type: EvidenceProjectionType,
    batch: tuple[EvidenceRecord, ...],
) -> dict[str, object]:
    requirements = scoped_projection_requirements(request.plan, request.projection_requirement_ids)
    return {
        "question": request.question,
        "projection_type": projection_type.value,
        "projection_requirement_ids": [r.requirement_id for r in requirements],
        "projection_requirements": [
            {
                "requirement_id": requirement.requirement_id,
                "description": requirement.description,
                "projection_type": projection_type.value,
            }
            for requirement in requirements
        ],
        "evidence": [_payload(item, request.plan) for item in batch],
    }


def _payload(record: EvidenceRecord, plan: ResearchPlan) -> dict[str, object]:
    source = record.evidence
    return {
        "evidence_id": record.evidence_id,
        "existing_verified_requirement_ids": list(verified_support_ids(record, plan.requirements)),
        "source_spans": source_spans_payload(source.content),
        "kind": source.source_type.value,
        "paper_id": source.paper_id,
        "paper_title": source.paper_title,
        "section_id": source.section_id,
        "section": source.section,
        "page_start": source.page_start,
        "page_end": source.page_end,
        "source_chunk_ids": list(source.source_chunk_ids),
        "seed_chunk_ids": list(source.seed_chunk_ids),
    }


def _failed(batch: tuple[EvidenceRecord, ...]) -> tuple[EvidenceAnalysis, ...]:
    return tuple(
        EvidenceAnalysis(
            item.evidence_id, EvidenceAnalysisStatus.FAILED,
            error_code="EvidenceProjectionFailed",
        )
        for item in batch
    )


def _convert(
    records: tuple[EvidenceRecord, ...], request: ProjectionRequest,
    projection_type: EvidenceProjectionType, draft: EvidenceProjectionBatchDraft,
) -> tuple[EvidenceAnalysis, ...]:
    expected = {record.evidence_id for record in records}
    actual = [item.evidence_id for item in draft.projections]
    if set(actual) != expected:
        raise ValueError("Evidence Projector 必须完整返回输入 evidence_id。")
    by_id = {item.evidence_id: item for item in records}
    known = {item.requirement_id for item in request.plan.requirements}
    grouped: dict[str, list[object]] = {evidence_id: [] for evidence_id in expected}
    for item in draft.projections:
        if item.projection_type != projection_type.value:
            raise ValueError("Projector 返回的 projection_type 与当前任务不一致。")
        grouped[item.evidence_id].append(item)
    converted = []
    for evidence_id, items in grouped.items():
        if len(items) > 1 and projection_type is not EvidenceProjectionType.FACT:
            raise ValueError("同一 Evidence 仅 Fact 类型可返回多条投影。")
        record = by_id[evidence_id]
        allowed = set(request.projection_requirement_ids).intersection(known)
        projections = []
        all_rejections: list[str] = []
        qualified_items = [item for item in items if item.status == "qualified"]
        if not qualified_items:
            for item in items:
                _validate_irrelevant(item)
            candidate = EvidenceAnalysis(evidence_id, EvidenceAnalysisStatus.IRRELEVANT,
                evaluated_requirement_ids=request.projection_requirement_ids)
            converted.append(candidate)
            continue
        for item in qualified_items:
            reported_supported = tuple(dict.fromkeys(item.supported_requirement_ids))
            if not set(reported_supported).issubset(allowed):
                raise ValueError("supported_requirement_ids 必须属于当前 projection Requirement scope。")
            projection, rejections = _projection(
                item, reported_supported, allowed, request.plan, record
            )
            projections.append(projection)
            all_rejections.extend(rejections)
        typed_scope = all(r.projection_type is not None for r in request.plan.requirements
                          if r.requirement_id in allowed)
        has_verified_support = any(projection.supported_requirement_ids for projection in projections)
        if (typed_scope and not has_verified_support
                and any(reason.startswith("support_rejected:") for reason in all_rejections)):
            raise ProjectionIntegrityError(evidence_id, allowed, all_rejections)
        if projection_type is EvidenceProjectionType.FACT:
            projection = FactProjection(
                evidence_id,
                tuple(fact for part in projections for fact in part.facts),
            )
        else:
            projection = projections[0]
        candidate = EvidenceAnalysis(
            evidence_id=evidence_id,
            status=EvidenceAnalysisStatus.QUALIFIED,
            projections=(projection,),
            projection_rejections=tuple(dict.fromkeys(all_rejections)),
            evaluated_requirement_ids=request.projection_requirement_ids,
        )
        converted.append(candidate)
    return tuple(converted)


def _validate_irrelevant(item: object) -> None:
    values = getattr(item, "supported_requirement_ids", [])
    if values:
        raise ValueError("irrelevant Projection 不得支持 Requirement。")
    if getattr(item, "support_candidates", []):
        raise ValueError("irrelevant Projection 不得提出支持锚点。")
    for field in ("entities", "comparison_dimensions", "steps", "inputs", "outputs"):
        if getattr(item, field, []) not in ([], ()):
            raise ValueError("irrelevant Projection 必须为空。")
    for field in ("fact_type", "value", "ownership", "mechanism"):
        if hasattr(item, field) and getattr(item, field) is not None:
            raise ValueError("irrelevant Projection 必须为空。")


def _projection(
    item: object,
    reported_supported: tuple[str, ...],
    allowed: set[str],
    plan: ResearchPlan,
    record: EvidenceRecord,
):
    if isinstance(item, EnumerationProjectionDraft):
        entities, rejections = _validated_enumeration(
            item.entities, allowed, plan, evidence_content=record.evidence.content
        )
        verified, anchors, anchor_rejections = _validated_support_candidates(
            item, reported_supported, allowed, plan, record
        )
        rejections.extend(anchor_rejections)
        # The Projector, not a keyword-to-role rule, decides which requirement
        # the retained entities support. Unknown ownership cannot ground it.
        supported = verified if any(
            entity_attribution_valid(entity.role, entity.ownership)
            for entity in entities
        ) else ()
        for requirement_id in verified:
            if requirement_id not in supported:
                rejections.append(f"support_rejected:{requirement_id}:no_attributed_entity")
        return EnumerationProjection(
            item.evidence_id, tuple(entities), supported,
            tuple(anchor for anchor in anchors if anchor.requirement_id in supported),
        ), tuple(rejections)
    verified, anchors, rejections = _validated_support_candidates(
        item, reported_supported, allowed, plan, record
    )
    if isinstance(item, ComparisonProjectionDraft):
        return ComparisonProjection(
            item.evidence_id, tuple(item.comparison_dimensions), verified,
            reported_supported, anchors,
        ), rejections
    if isinstance(item, FactProjectionDraft):
        if item.fact_type is None or item.value is None or item.ownership is None:
            raise ValueError("qualified FactProjection requires fact_type, value and ownership.")
        return FactProjection(item.evidence_id, (
            FactItem(
                item.fact_type, item.value, item.ownership,
                verified, reported_supported, anchors,
            ),
        )), rejections
    if isinstance(item, MechanismProjectionDraft):
        if item.mechanism is None:
            raise ValueError("qualified MechanismProjection requires mechanism.")
        return MechanismProjection(
            item.evidence_id, item.mechanism, tuple(item.steps),
            tuple(item.inputs), tuple(item.outputs), verified,
            reported_supported, anchors,
        ), rejections
    raise ValueError("Unknown projection draft.")


def _validated_support_candidates(
    item: object,
    proposed: tuple[str, ...],
    allowed: set[str],
    plan: ResearchPlan,
    record: EvidenceRecord,
) -> tuple[tuple[str, ...], tuple[RequirementSupportAnchor, ...], tuple[str, ...]]:
    by_id = {requirement.requirement_id: requirement for requirement in plan.requirements}
    candidates = getattr(item, "support_candidates", ())
    verified: list[str] = []
    anchors: list[RequirementSupportAnchor] = []
    rejections: list[str] = []
    for requirement_id in proposed:
        if requirement_id not in allowed:
            continue
        matching = [candidate for candidate in candidates if candidate.requirement_id == requirement_id]
        if not matching:
            rejections.append(f"support_rejected:{requirement_id}:missing_source_anchor")
            continue
        accepted = False
        for candidate in matching:
            try:
                source_excerpt = resolve_source_span_ids(
                    record.evidence.content, candidate.source_span_ids
                )
            except SourceSpanResolutionError as error:
                rejections.append(
                    f"support_rejected:{requirement_id}:{error.code}"
                )
                continue
            anchor = RequirementSupportAnchor(
                requirement_id,
                local_source_window(record.evidence.content, source_excerpt)
                if isinstance(item, FactProjectionDraft)
                else source_excerpt,
                candidate.support_scope,
                tuple(candidate.unresolved_aspects),
            )
            reason = support_rejection(
                requirement=by_id[requirement_id], record=record,
                anchor=anchor,
                fact_ownership=(
                    item.ownership if isinstance(item, FactProjectionDraft) else None
                ),
            )
            if reason is None:
                verified.append(requirement_id)
                anchors.append(anchor)
                accepted = True
                break
            rejections.append(f"support_rejected:{requirement_id}:{reason}")
            rejections.append(
                f"support_rejection_detail:{requirement_id}:code={reason}:"
                f"anchor_length={len(anchor.source_excerpt)}"
            )
        if not accepted:
            continue
    for candidate in candidates:
        if candidate.requirement_id not in proposed:
            rejections.append(
                f"support_rejected:{candidate.requirement_id}:not_proposed"
            )
    return tuple(verified), tuple(anchors), tuple(dict.fromkeys(rejections))


def _revalidate_cached_single(
    record: EvidenceRecord, analysis: EvidenceAnalysis, plan: ResearchPlan
) -> EvidenceAnalysis:
    projection, = analysis.projections  # wrapper explicitly supplies one group projection
    if isinstance(projection, EnumerationProjection):
        allowed = {item.requirement_id for item in plan.requirements}
        entities, rejections = _validated_enumeration(
            projection.entities, allowed, plan, evidence_content=record.evidence.content
        )
        anchors = tuple(
            anchor for anchor in projection.support_anchors
            if anchor.requirement_id in allowed
            and source_contains(record.evidence.content, anchor.source_excerpt)
        )
        verified = tuple(
            requirement_id for requirement_id in projection.supported_requirement_ids
            if requirement_id in allowed and any(
                entity_attribution_valid(entity.role, entity.ownership)
                for entity in entities
            )
            and any(anchor.requirement_id == requirement_id for anchor in anchors)
        )
        rejections.extend(
            f"support_rejected:{requirement_id}:cache_integrity_revalidation"
            for requirement_id in projection.supported_requirement_ids
            if requirement_id not in verified
        )
        if (
            tuple(entities) == projection.entities
            and verified == projection.supported_requirement_ids
            and anchors == projection.support_anchors
            and not rejections
        ):
            return analysis
        return replace(
            analysis,
            projection=None, projections=(EnumerationProjection(record.evidence_id, tuple(entities), verified, anchors),),
            projection_rejections=tuple(dict.fromkeys((*analysis.projection_rejections, *rejections))),
        )
    candidate_record = replace(record, assessment=analysis)
    if isinstance(projection, FactProjection):
        retained = []
        rejected = []
        for fact in projection.facts:
            single = replace(analysis, projection=None, projections=(FactProjection(record.evidence_id, (fact,)),))
            verified = verified_support_ids(
                replace(record, assessment=single), plan.requirements
            )
            retained.append(replace(fact, supported_requirement_ids=verified))
            rejected.extend(
                f"support_rejected:{requirement_id}:cache_integrity_revalidation"
                for requirement_id in fact.supported_requirement_ids
                if requirement_id not in verified
            )
        if not rejected:
            return analysis
        return replace(
            analysis,
            projection=None, projections=(FactProjection(record.evidence_id, tuple(retained)),),
            projection_rejections=(*analysis.projection_rejections, *rejected),
        )
    verified = verified_support_ids(candidate_record, plan.requirements)
    if verified == projection.supported_requirement_ids:
        return analysis
    rejected = tuple(
        f"support_rejected:{requirement_id}:cache_integrity_revalidation"
        for requirement_id in projection.supported_requirement_ids
        if requirement_id not in verified
    )
    return replace(
        analysis,
        projection=None, projections=(replace(projection, supported_requirement_ids=verified),),
        projection_rejections=(*analysis.projection_rejections, *rejected),
    )


def _validated_enumeration(
    drafts: Iterable[object],
    allowed_requirement_ids: set[str],
    plan: ResearchPlan,
    *,
    evidence_content: str,
) -> tuple[list[ProjectedEntity], list[str]]:
    """Check entity source presence and structural attribution only."""
    accepted: list[ProjectedEntity] = []
    rejections: list[str] = []
    seen: set[tuple[str, str, str, str]] = set()
    source_text = _normalized_text(evidence_content)
    for draft in drafts:
        name = str(getattr(draft, "canonical_name", "")).strip()
        if not name:
            rejections.append("entity_rejected:<empty>:reason=empty_canonical_name")
            continue
        admitted_name = _name_present_in_evidence(name, source_text)
        if admitted_name is None:
            entity_type = str(getattr(draft, "entity_type", ""))
            role = str(getattr(draft, "role", "unknown"))
            ownership = str(getattr(draft, "ownership", "unknown"))
            rejections.append(
                f"entity_rejected:{name}:reason=not_present_in_evidence:"
                f"role={role}:ownership={ownership}:entity_type={entity_type}"
            )
            continue
        entity = ProjectedEntity(
            admitted_name,
            str(getattr(draft, "entity_type", "")),
            str(getattr(draft, "role", "unknown")),
            str(getattr(draft, "ownership", "unknown")),
        )
        if (entity.role is not ProjectedEntityRole.UNKNOWN
            and entity.ownership is not EvidenceOwnership.UNKNOWN
            and not entity_attribution_valid(entity.role, entity.ownership)):
            rejections.append(
                f"entity_rejected:{entity.canonical_name}:reason=invalid_attribution"
            )
            continue
        key = (
            entity.canonical_name.casefold(), entity.entity_type,
            entity.role.value, entity.ownership.value,
        )
        if key in seen:
            rejections.append(
                f"entity_rejected:{entity.canonical_name}:reason=duplicate:"
                f"role={entity.role.value}:ownership={entity.ownership.value}:"
                f"entity_type={entity.entity_type}"
            )
            continue
        seen.add(key)
        accepted.append(entity)
    return accepted, rejections


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _name_present_in_evidence(name: str, source_text: str) -> str | None:
    """Return a text-grounded canonical name, allowing an overlong LLM label.

    Parsed papers frequently contain expanded architecture descriptions.  When
    the leading entity head (at least three words) occurs verbatim in Evidence,
    preserve that grounded head instead of deleting the entity wholesale.
    """
    original_words = " ".join(name.split()).split()
    normalized = " ".join(word.casefold() for word in original_words)
    if normalized in source_text:
        return " ".join(original_words)
    for count in range(len(original_words) - 1, 2, -1):
        prefix = " ".join(original_words[:count])
        if prefix.casefold() in source_text:
            return prefix
    return None




def _revalidate_cached(record, analysis, plan):
    if not analysis.projections:
        return analysis
    validated, rejections = [], list(analysis.projection_rejections)
    for projection in analysis.projections:
        single = replace(analysis, projection=None, projections=(projection,))
        result = _revalidate_cached_single(record, single, plan)
        validated.extend(result.projections)
        rejections.extend(result.projection_rejections)
    return replace(analysis, projection=None, projections=tuple(validated),
                   projection_rejections=tuple(dict.fromkeys(rejections)))
