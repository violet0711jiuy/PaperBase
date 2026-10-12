"""Deterministic Action control-plane normalization and safe diagnostics helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .contracts import ActionCanonicalization, CanonicalizationOperation


# These fields belong to a retired Action contract.  They do not carry a
# current runtime decision and are explicitly removed for backward-compatible
# model outputs.  Other unknown fields remain visible to Pydantic's strict
# schema and therefore fail validation.
_HARMLESS_DEPRECATED_FIELDS = frozenset({
    "target_subtask_id",
    "missing_information",
    # Retrieval routing is owned by the bound Plan Requirement.  Accept and
    # discard this retired model field so saved/slow-provider responses remain
    # recoverable without granting it semantic authority.
    "evidence_targets",
})


class ActionCanonicalizer:
    """Apply only deterministic, non-semantic repairs before schema validation.

    It never invents or drops IDs, chooses a Tool, changes an Action name, or
    rewrites a query.  The caller must still run strict schema, domain, and
    runtime Guard validation afterwards.
    """

    def canonicalize(self, payload: object) -> tuple[object, ActionCanonicalization]:
        if not isinstance(payload, dict):
            return payload, ActionCanonicalization()
        result: dict[str, Any] = deepcopy(payload)
        decision = result.get("decision")
        if not isinstance(decision, dict):
            return result, ActionCanonicalization()

        operations: list[CanonicalizationOperation] = []
        self._remove_deprecated_fields(decision, "decision", operations)
        action_name = decision.get("action")
        if action_name == "search_papers":
            self._strip_identifier(
                decision, "requirement_id", "requirement_id", operations
            )
            searches = decision.get("searches")
            if isinstance(searches, list):
                for index, search in enumerate(searches):
                    if isinstance(search, dict):
                        self._normalize_search(
                            search, f"searches[{index}]", operations
                        )
        elif action_name in {
            "search_knowledge_base",
            "search_references",
        }:
            self._strip_identifier(
                decision, "requirement_id", "requirement_id", operations
            )
            self._normalize_search(decision, "decision", operations)
        elif action_name == "get_section_context":
            self._strip_identifier(decision, "paper_id", "paper_id", operations)
            self._strip_identifier(decision, "section_id", "section_id", operations)
            self._strip_identifier(decision, "requirement_id", "requirement_id", operations)
        elif action_name == "get_paper_overview":
            self._strip_identifier(decision, "paper_id", "paper_id", operations)
            self._strip_identifier(decision, "requirement_id", "requirement_id", operations)

        return result, ActionCanonicalization(
            changed=bool(operations),
            changed_fields=tuple(dict.fromkeys(item.field_path for item in operations)),
            operations=tuple(operations),
        )

    @staticmethod
    def _record(
        operations: list[CanonicalizationOperation],
        field_path: str,
        operation: str,
        count: int,
    ) -> None:
        if count:
            operations.append(CanonicalizationOperation(field_path, operation, count))

    def _remove_deprecated_fields(
        self,
        value: dict[str, Any],
        field_path: str,
        operations: list[CanonicalizationOperation],
    ) -> None:
        removed = 0
        for name in _HARMLESS_DEPRECATED_FIELDS:
            if name in value:
                del value[name]
                removed += 1
        self._record(operations, field_path, "removed_deprecated_field", removed)

    def _normalize_search(
        self,
        value: dict[str, Any],
        prefix: str,
        operations: list[CanonicalizationOperation],
    ) -> None:
        self._remove_deprecated_fields(value, prefix, operations)
        self._strip_identifier(value, "paper_id", f"{prefix}.paper_id", operations)
        self._normalize_semantic_query(value, f"{prefix}.semantic_query", operations)
        self._normalize_lexical_terms(value, f"{prefix}.lexical_terms", operations)

    def _strip_identifier(
        self,
        value: dict[str, Any],
        name: str,
        field_path: str,
        operations: list[CanonicalizationOperation],
    ) -> None:
        original = value.get(name)
        if isinstance(original, str):
            normalized = original.strip()
            if normalized != original:
                value[name] = normalized
                self._record(operations, field_path, "trimmed", 1)

    def _normalize_semantic_query(
        self,
        value: dict[str, Any],
        field_path: str,
        operations: list[CanonicalizationOperation],
    ) -> None:
        original = value.get("semantic_query")
        if isinstance(original, str):
            normalized = " ".join(original.split())
            if normalized != original:
                value["semantic_query"] = normalized
                self._record(operations, field_path, "normalized_whitespace", 1)

    def _normalize_lexical_terms(
        self,
        value: dict[str, Any],
        field_path: str,
        operations: list[CanonicalizationOperation],
    ) -> None:
        original = value.get("lexical_terms")
        if not isinstance(original, list):
            return
        trimmed = [item.strip() if isinstance(item, str) else item for item in original]
        trim_count = sum(
            isinstance(before, str) and before != after
            for before, after in zip(original, trimmed)
        )
        non_empty = [item for item in trimmed if not (isinstance(item, str) and not item)]
        self._record(operations, field_path, "trimmed", trim_count)
        self._record(operations, field_path, "removed_empty", len(trimmed) - len(non_empty))
        deduplicated: list[Any] = []
        seen: set[tuple[str, str] | tuple[str, int]] = set()
        for item in non_empty:
            key: tuple[str, str] | tuple[str, int]
            if isinstance(item, str):
                key = ("text", item.casefold())
            else:
                key = ("identity", id(item))
            if key not in seen:
                seen.add(key)
                deduplicated.append(item)
        self._record(operations, field_path, "deduplicated", len(non_empty) - len(deduplicated))
        # Schema validates the five-term limit.  Silently keeping the first
        # five would discard later, more discriminative terms before repair.
        value["lexical_terms"] = deduplicated
