"""运行期证据注册、稳定编号与精确去重。"""

from __future__ import annotations

from dataclasses import replace

from paperbase.retrieval.contracts import (
    SearchPapersPayload,
    ToolResult,
)

from .contracts import (
    EvidenceAnalysis,
    EvidenceQualification,
    EvidenceRecord,
    RegistryDelta,
    RegistryState,
)


class EvidenceRegistry:
    """按来源、论文和 source chunk 集合做 exact dedup。"""

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
        iteration: int,
        supporting_queries: tuple[str, ...] = (),
        target_subtask_ids: tuple[str, ...] = (),
    ) -> RegistryDelta:
        added: list[str] = []
        reused: list[str] = []
        for evidence, query, targets, missing, subtask_id in _evidence_with_queries(
            result, supporting_queries
        ):
            key = _evidence_key(evidence)
            current = self._records_by_key.get(key)
            query_values = tuple(
                dict.fromkeys(
                    value.strip()
                    for value in (*supporting_queries, query)
                    if value and value.strip()
                )
            )
            if current is not None:
                merged_queries = tuple(dict.fromkeys((*current.supporting_queries, *query_values)))
                self._records_by_key[key] = replace(
                    current,
                    supporting_queries=merged_queries,
                    last_seen_iteration=iteration,
                    originating_tools=tuple(
                        dict.fromkeys((*current.originating_tools, result.tool_name.value))
                    ),
                    target_subtask_ids=tuple(
                        dict.fromkeys(
                            (
                                *current.target_subtask_ids,
                                *target_subtask_ids,
                                *((subtask_id,) if subtask_id else ()),
                            )
                        )
                    ),
                    requested_evidence_targets=frozenset(
                        (*current.requested_evidence_targets, *targets)
                    ),
                    requested_missing_information=tuple(
                        dict.fromkeys(
                            (*current.requested_missing_information, *missing)
                        )
                    ),
                )
                reused.append(current.evidence_id)
                continue
            if evidence.source_type.value == "bibliography":
                self._reference_counter += 1
                evidence_id = f"R{self._reference_counter}"
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
                target_subtask_ids=tuple(dict.fromkeys(target_subtask_ids)),
                requested_evidence_targets=frozenset(targets),
                requested_missing_information=tuple(missing),
            )
            if subtask_id:
                current_record = self._records_by_key[key]
                self._records_by_key[key] = replace(
                    current_record,
                    target_subtask_ids=tuple(
                        dict.fromkeys((*current_record.target_subtask_ids, subtask_id))
                    ),
                )
            added.append(evidence_id)
        return RegistryDelta(tuple(added), tuple(reused))

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return next(
            (record for record in self._records_by_key.values() if record.evidence_id == evidence_id),
            None,
        )

    def apply_qualifications(
        self, qualifications: tuple[EvidenceQualification, ...]
    ) -> tuple[str, ...]:
        """把 Agent 标注写回已登记证据；拒绝未知或重复的 Evidence ID。"""
        evidence_ids = [item.evidence_id for item in qualifications]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("同一轮 Qualification 不能重复标注同一 Evidence。")
        records_by_id = {record.evidence_id: record for record in self._records_by_key.values()}
        unknown_ids = tuple(
            evidence_id for evidence_id in evidence_ids if evidence_id not in records_by_id
        )
        if unknown_ids:
            raise ValueError("Qualification 引用了未知 Evidence：" + ", ".join(unknown_ids))
        for qualification in qualifications:
            current = records_by_id[qualification.evidence_id]
            key = _evidence_key(current.evidence)
            self._records_by_key[key] = replace(current, qualification=qualification)
        return tuple(evidence_ids)

    def apply_analyses(
        self, analyses: tuple[EvidenceAnalysis, ...]
    ) -> tuple[str, ...]:
        """把独立 Evidence Analysis 原子写回 Registry。"""
        evidence_ids = [item.evidence_id for item in analyses]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("同一轮 Evidence Analysis 不能重复处理同一 Evidence。")
        records_by_id = {
            record.evidence_id: record for record in self._records_by_key.values()
        }
        unknown_ids = tuple(
            evidence_id
            for evidence_id in evidence_ids
            if evidence_id not in records_by_id
        )
        if unknown_ids:
            raise ValueError(
                "Evidence Analysis 引用了未知 Evidence：" + ", ".join(unknown_ids)
            )
        for analysis in analyses:
            current = records_by_id[analysis.evidence_id]
            key = _evidence_key(current.evidence)
            self._records_by_key[key] = replace(
                current,
                analysis_status=analysis.status,
                digest=analysis.digest,
                claims=analysis.claims,
                analysis_error=analysis.error_code,
                qualification=None,
            )
        return tuple(evidence_ids)

    def snapshot(self) -> RegistryState:
        return RegistryState(
            records=self.records,
            content_counter=self._content_counter,
            reference_counter=self._reference_counter,
        )


def _evidence_with_queries(
    result: ToolResult, fallback_queries: tuple[str, ...]
):
    payload = result.payload
    if isinstance(payload, SearchPapersPayload):
        for paper_result in payload.paper_results:
            for evidence in paper_result.evidence_candidates:
                yield (
                    evidence,
                    paper_result.query,
                    paper_result.evidence_targets,
                    paper_result.missing_information,
                    paper_result.target_subtask_id,
                )
        return
    query = getattr(payload, "query", "")
    for evidence in getattr(payload, "evidence_candidates", ()):
        yield (
            evidence,
            query or (fallback_queries[0] if fallback_queries else ""),
            frozenset(),
            (),
            None,
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
