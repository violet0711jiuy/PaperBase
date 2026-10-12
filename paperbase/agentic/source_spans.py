"""Evidence-local span utilities for offline diagnostics and historical replay.

The provider selects opaque span IDs.  Only this module turns those IDs back
into exact source text, so copied/paraphrased provider excerpts never become
the provenance authority. Current Agentic Answer uses intact Evidence-level
citations and does not call this module or expose these spans to its model.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_CAPTION_PREFIX = re.compile(r"^(?:fig(?:ure)?\.?|table|图|表)\s*\d*", re.IGNORECASE)
_SENTENCE_END = re.compile(
    r"(?:[。！？!?](?=\s|$)|\.(?=\s+(?:[A-Z0-9\[\(\"'])|$))"
)
_DISPLAY_MATH = re.compile(r"(?:\\begin\{|\\frac|\\sum|\\int|\$\$|\\\[|\\\])")
_MAX_ENVELOPE_CHARS = 2400
_MAX_ENVELOPE_SPANS = 16
_MAX_CITATION_RUNS = 8


@dataclass(frozen=True)
class SourceSpan:
    span_id: str
    start: int
    end: int
    span_type: str
    text: str

    def as_prompt_payload(self) -> dict[str, object]:
        return {
            "span_id": self.span_id,
            "start": self.start,
            "end": self.end,
            "span_type": self.span_type,
            "text": self.text,
        }


class SourceSpanResolutionError(ValueError):
    """A safe, machine-readable span selection failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def build_source_spans(content: str) -> tuple[SourceSpan, ...]:
    """Split source text without changing characters or offsets.

    Non-empty lines are the structural boundary.  Prose lines are additionally
    split at conservative sentence boundaries; tables, captions, and equations
    stay intact.  Very long unpunctuated lines are split only at whitespace.
    """
    ranges: list[tuple[int, int]] = []
    for line_match in re.finditer(r"[^\r\n]+", content):
        raw_start, raw_end = line_match.span()
        line = line_match.group(0)
        left = len(line) - len(line.lstrip())
        right = len(line.rstrip())
        if left >= right:
            continue
        start, end = raw_start + left, raw_start + right
        text = content[start:end]
        if _span_type(text) != "sentence":
            ranges.append((start, end))
            continue
        ranges.extend(_prose_ranges(content, start, end))
    return tuple(
        SourceSpan(f"S{index}", start, end, _span_type(content[start:end]), content[start:end])
        for index, (start, end) in enumerate(ranges, start=1)
    )


def source_spans_payload(content: str) -> list[dict[str, object]]:
    return [span.as_prompt_payload() for span in build_source_spans(content)]


def resolve_source_span_ids(content: str, span_ids: Iterable[str]) -> str:
    """Resolve selected IDs to a bounded, exact source envelope.

    Academic support often spans a definition, an equation, and a later
    explanation.  The provider therefore selects the relevant atoms while the
    program owns ordering and restores the smallest enclosing source slice.
    Very broad selections remain invalid so an anchor cannot silently become
    an entire Evidence chunk.
    """
    requested = tuple(span_ids)
    if not requested:
        raise SourceSpanResolutionError("source_span_missing")
    if len(set(requested)) != len(requested):
        raise SourceSpanResolutionError("source_spans_duplicate")
    spans = build_source_spans(content)
    by_id = {span.span_id: (index, span) for index, span in enumerate(spans)}
    if any(span_id not in by_id for span_id in requested):
        raise SourceSpanResolutionError("source_span_not_found")
    indices = [by_id[span_id][0] for span_id in requested]
    if indices != sorted(indices):
        raise SourceSpanResolutionError("source_spans_out_of_order")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise SourceSpanResolutionError("source_spans_non_contiguous")
    first = spans[indices[0]]
    last = spans[indices[-1]]
    if (
        indices[-1] - indices[0] + 1 > _MAX_ENVELOPE_SPANS
        or last.end - first.start > _MAX_ENVELOPE_CHARS
    ):
        raise SourceSpanResolutionError("source_spans_too_broad")
    return content[first.start:last.end]


def partition_source_span_ids(
    content: str, span_ids: Iterable[str]
) -> tuple[tuple[str, ...], ...]:
    """Split an ordered selection into independently contiguous anchors.

    A model may select two relevant passages from the same Evidence while
    omitting unrelated sentences between them.  Expanding to the enclosing
    range would silently cite those sentences, while rejecting the answer
    loses otherwise valid grounding.  This function preserves the exact IDs
    and their order, splitting only at gaps.  Unknown, duplicate, or
    out-of-order IDs remain hard failures.
    """
    requested = tuple(span_ids)
    if not requested:
        raise SourceSpanResolutionError("source_span_missing")
    if len(set(requested)) != len(requested):
        raise SourceSpanResolutionError("source_spans_duplicate")
    spans = build_source_spans(content)
    by_id = {span.span_id: index for index, span in enumerate(spans)}
    if any(span_id not in by_id for span_id in requested):
        raise SourceSpanResolutionError("source_span_not_found")
    indices = [by_id[span_id] for span_id in requested]
    if indices != sorted(indices):
        raise SourceSpanResolutionError("source_spans_out_of_order")

    runs: list[list[str]] = [[requested[0]]]
    for previous, current, span_id in zip(indices, indices[1:], requested[1:]):
        if current == previous + 1:
            runs[-1].append(span_id)
        else:
            runs.append([span_id])
    if len(runs) > _MAX_CITATION_RUNS:
        raise SourceSpanResolutionError("source_spans_too_fragmented")
    result = tuple(tuple(run) for run in runs)
    # Reuse the canonical envelope checks for each independently cited run.
    for run in result:
        resolve_source_span_ids(content, run)
    return result


def source_span_ids_for_excerpt(content: str, excerpt: str) -> tuple[str, ...]:
    """Map an exact source slice to spans; useful for deterministic adapters/tests."""
    start = content.find(excerpt)
    if start < 0 or not excerpt:
        raise SourceSpanResolutionError("source_span_not_found")
    end = start + len(excerpt)
    selected = tuple(
        span.span_id for span in build_source_spans(content)
        if span.start < end and span.end > start
    )
    if not selected:
        raise SourceSpanResolutionError("source_span_not_found")
    return selected


def _prose_ranges(content: str, start: int, end: int) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    cursor = start
    for boundary in _SENTENCE_END.finditer(content, start, end):
        stop = boundary.end()
        if stop <= cursor:
            continue
        ranges.extend(_bounded_range(content, cursor, stop))
        cursor = stop
        while cursor < end and content[cursor].isspace():
            cursor += 1
    if cursor < end:
        ranges.extend(_bounded_range(content, cursor, end))
    return ranges


def _bounded_range(content: str, start: int, end: int, *, limit: int = 1200) -> list[tuple[int, int]]:
    result: list[tuple[int, int]] = []
    while end - start > limit:
        cut = content.rfind(" ", start, start + limit + 1)
        if cut <= start:
            cut = start + limit
        result.append((start, cut))
        start = cut
        while start < end and content[start].isspace():
            start += 1
    if start < end:
        result.append((start, end))
    return result


def _span_type(text: str) -> str:
    stripped = text.strip()
    if _CAPTION_PREFIX.match(stripped):
        return "caption"
    if stripped.count("|") >= 2 or "\t" in stripped:
        return "table_row"
    if _DISPLAY_MATH.search(stripped) or (
        "=" in stripped and len(stripped) <= 240 and not re.search(r"[。！？!?]", stripped)
    ):
        return "equation"
    return "sentence"
