from __future__ import annotations

import pytest

from paperbase.agentic.source_spans import (
    SourceSpanResolutionError,
    build_source_spans,
    resolve_source_span_ids,
)


def test_source_spans_preserve_exact_offsets_and_types() -> None:
    content = "Figure 1 Model overview\nloss = y - pred\n| A | B |\nFirst fact. Second fact."
    spans = build_source_spans(content)
    assert [span.span_type for span in spans] == [
        "caption", "equation", "table_row", "sentence", "sentence",
    ]
    assert all(content[span.start:span.end] == span.text for span in spans)
    assert [span.span_id for span in spans] == ["S1", "S2", "S3", "S4", "S5"]


def test_contiguous_span_ids_restore_one_exact_source_slice() -> None:
    content = "First fact. Second fact. Third fact."
    assert resolve_source_span_ids(content, ("S1", "S2")) == "First fact. Second fact."


def test_sparse_or_out_of_order_span_ids_restore_a_bounded_canonical_envelope() -> None:
    content = "First fact. Second fact. Third fact. Fourth fact."
    assert resolve_source_span_ids(content, ("S3", "S1")) == (
        "First fact. Second fact. Third fact."
    )


@pytest.mark.parametrize(
    ("span_ids", "code"),
    (
        (("S9",), "source_span_not_found"),
        (("S1", "S1"), "source_spans_duplicate"),
    ),
)
def test_invalid_span_combinations_have_repairable_codes(span_ids, code) -> None:
    with pytest.raises(SourceSpanResolutionError) as caught:
        resolve_source_span_ids("First fact. Second fact. Third fact.", span_ids)
    assert caught.value.code == code


def test_sparse_span_envelope_is_rejected_when_it_is_too_broad() -> None:
    content = " ".join(f"Sentence {index} has enough filler text to be distinct." for index in range(30))
    with pytest.raises(SourceSpanResolutionError) as caught:
        resolve_source_span_ids(content, ("S1", "S30"))
    assert caught.value.code == "source_spans_too_broad"
