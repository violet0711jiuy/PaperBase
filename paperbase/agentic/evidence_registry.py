"""运行期证据注册、稳定编号与精确去重。"""

from __future__ import annotations

from dataclasses import replace

from paperbase.retrieval.contracts import (
    SearchPapersPayload,
    ToolResult,
)

from .contracts import (
    AgentAction,
    EvidenceRecord,
    RegistryDelta,
    RegistryState,
    ResearchPlan,
    SearchPapersAction,
)

from .requirement_targets import targets_for_requirement_ids, requirement_target_bindings


class EvidenceRegistry:
    """按来源身份去重，只跨轮复用已完全覆盖的扩展窗口。"""

    def __init__(self, state: RegistryState | None = None) -> None:
        snapshot = state or RegistryState()
        self._records_by_key = {
            _evidence_key(record.evidence): record for record in snapshot.records
        }
        self._content_counter = snapshot.content_counter
        self._reference_counter = snapshot.reference_counter

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records_by_key.values())

    def register_tool_result(
        self,
        result: ToolResult,
        *,
        action: AgentAction,
        iteration: int,
        supporting_queries: tuple[str, ...] = (),
        plan: ResearchPlan | None = None,
    ) -> RegistryDelta:
        if plan is not None:
            requirement_target_bindings(plan, action)  # validate all before Registry mutation
        added: list[str] = []
        reused: list[str] = []
        for evidence, query, targets, requirement_ids in _evidence_with_queries(
            result, action, supporting_queries, plan
        ):
            key = _evidence_key(evidence)
            storage_key = key
            current = self._records_by_key.get(key)
            if current is None:
                overlapping = self._find_overlapping_record(evidence)
                if overlapping is not None:
                    storage_key, current = overlapping
            query_values = tuple(
                dict.fromkeys(
                    value.strip()
                    for value in (*supporting_queries, query)
                    if value and value.strip()
                )
            )
            if current is not None:
                merged_queries = tuple(dict.fromkeys((*current.supporting_queries, *query_values)))
                self._records_by_key[storage_key] = replace(
                    current,
                    supporting_queries=merged_queries,
                    last_seen_iteration=iteration,
                    originating_tools=tuple(
                        dict.fromkeys((*current.originating_tools, result.tool_name.value))
                    ),
                    requested_evidence_targets=(
                        targets_for_requirement_ids(plan, tuple(
                            rid for rid in dict.fromkeys((*current.requested_requirement_ids, *requirement_ids))
                            if rid in {item.requirement_id for item in plan.requirements}
                        )) if plan is not None else frozenset((*current.requested_evidence_targets, *targets))
                    ),
                    requested_requirement_ids=tuple(
                        dict.fromkeys(
                            (*current.requested_requirement_ids, *requirement_ids)
                        )
                    ),
                )
                reused.append(current.evidence_id)
                continue
            if evidence.source_type.value == "bibliography":
                self._reference_counter += 1
                evidence_id = f"REF{self._reference_counter}"
            else:
                self._content_counter += 1
                evidence_id = f"E{self._content_counter}"
            self._records_by_key[key] = EvidenceRecord(
                evidence_id=evidence_id,
                evidence=evidence,
                supporting_queries=query_values,
                first_seen_iteration=iteration,
                last_seen_iteration=iteration,
                originating_tools=(result.tool_name.value,),
                requested_evidence_targets=frozenset(targets),
                requested_requirement_ids=tuple(requirement_ids),
            )
            added.append(evidence_id)
        return RegistryDelta(tuple(dict.fromkeys(added)), tuple(dict.fromkeys(reused)))

    def _find_overlapping_record(
        self, evidence: object
    ) -> tuple[tuple[object, ...], EvidenceRecord] | None:
        return next(
            (
                (key, record)
                for key, record in self._records_by_key.items()
                if _same_evidence_window(record.evidence, evidence)
            ),
            None,
        )

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return next(
            (record for record in self._records_by_key.values() if record.evidence_id == evidence_id),
            None,
        )

    def snapshot(self) -> RegistryState:
        return RegistryState(
            records=self.records,
            content_counter=self._content_counter,
            reference_counter=self._reference_counter,
        )


def _evidence_with_queries(
    result: ToolResult, action: AgentAction, fallback_queries: tuple[str, ...],
    plan: ResearchPlan | None = None,
):
    if result.tool_name.value != action.action_name:
        raise ValueError("ToolResult and Action must refer to the same tool.")
    payload = result.payload
    if isinstance(payload, SearchPapersPayload):
        if not isinstance(action, SearchPapersAction) or len(payload.paper_results) != len(action.searches):
            raise ValueError("search_papers outcomes must match the Action searches.")
        if any(
            (outcome.paper_id, outcome.query) != (search.paper_id, search.query)
            for outcome, search in zip(payload.paper_results, action.searches)
        ):
            raise ValueError("search_papers outcome does not match its Action search.")
        for paper_result, search in zip(payload.paper_results, action.searches):
            for evidence in paper_result.evidence_candidates:
                yield (
                    evidence,
                    search.query,
                    targets_for_requirement_ids(plan, search.requirement_ids)
                    if plan is not None else search.evidence_targets,
                    search.requirement_ids,
                )
        return
    query = getattr(payload, "query", "")
    for evidence in getattr(payload, "evidence_candidates", ()):
        yield (
            evidence,
            query or (fallback_queries[0] if fallback_queries else ""),
            targets_for_requirement_ids(plan, getattr(action, "requirement_ids", ()))
            if plan is not None else frozenset(),
            getattr(action, "requirement_ids", ()),
        )


def _evidence_key(evidence: object) -> tuple[object, ...]:
    base = (
        evidence.source_type.value,
        evidence.paper_id,
        tuple(sorted(evidence.source_chunk_ids)),
    )
    if evidence.source_type.value == "overview":
        return (*base, evidence.retrieval_metadata.get("overview_field"))
    return base


def _same_evidence_window(left: object, right: object) -> bool:
    if (
        left.source_type != right.source_type
        or left.paper_id != right.paper_id
        or (
            left.section_id
            and right.section_id
            and left.section_id != right.section_id
        )
    ):
        return False
    left_ids = set(left.source_chunk_ids)
    right_ids = set(right.source_chunk_ids)
    if not left_ids or not right_ids:
        return False
    # Reuse must not hide newly retrieved chunks. Keep existing records and
    # their citation spans immutable; an expanded or shifted window containing
    # any new chunk gets its own Evidence ID and triggers Answer reassessment.
    return right_ids <= left_ids
