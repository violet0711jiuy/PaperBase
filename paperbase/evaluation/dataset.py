"""Reuse frozen-Golden validation, without writing remapped data or touching SQLite."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import closing
from pathlib import Path
import sqlite3
import re

from eval.scripts.validate_golden_dataset import (
    validate_dataset, validate_evidence, load_jsonl, open_database_read_only,
    load_chunks, load_paper_ids, validate_sqlite_schema,
)
from .agentic import gold_chunk_ids, gold_evidence_alternatives
from .contracts import DatasetIssue, GoldenValidation


def validate_golden_dataset(records: Sequence[Mapping], *, database: Path | None = None,
                            paper_ids: set[str] | None = None, chunks: Mapping | None = None,
                            parse_errors: Sequence[Mapping] = ()) -> GoldenValidation:
    """Basic consistency plus authoritative chunk existence; missing DB is an input error."""
    checked = paper_ids is not None and chunks is not None
    issues = []
    try:
        schema = validate_dataset(records, parse_errors)
    except (ValueError, TypeError, KeyError) as error:
        return GoldenValidation(len(records), (DatasetIssue(None, 'malformed_schema', type(error).__name__),),
            tuple(sorted({k for row in records for k in row})), False)
    for error in schema['parse_errors']:
        issues.append(DatasetIssue(None, 'jsonl_parse_error', str(error.get('message'))))
    for row in records:
        case_id=row.get('id')
        if isinstance(case_id,str) and not re.fullmatch(r'[A-Za-z0-9_-]+',case_id):
            issues.append(DatasetIssue(case_id,'invalid_case_id','Case ID is not a safe artifact filename'))
    for case_id in schema['duplicate_ids']:
        issues.append(DatasetIssue(case_id, 'duplicate_id', 'ID is not unique'))
    for key in ('missing_fields', 'invalid_primary_types', 'illegal_tags'):
        for case_id, detail in schema[key].items():
            issues.append(DatasetIssue(case_id, key, str(detail)))
    for key in ('schema_errors', 'logic_errors'):
        issues.extend(DatasetIssue(e.get('case_id'), key, e['message']) for e in schema[key])
    if database is not None:
        try:
            with closing(open_database_read_only(database)) as connection:
                if not validate_sqlite_schema(connection)['valid']:
                    raise ValueError('SQLite chunk/documents schema is not valid')
                paper_ids = load_paper_ids(connection)
                chunks = load_chunks(connection, (
                    chunk_id
                    for record in records
                    for chunk_id in (
                        *gold_chunk_ids(record),
                        *(
                            alternative
                            for paths in gold_evidence_alternatives(record).values()
                            for path in paths
                            for alternative in path
                        ),
                    )
                ))
                checked = True
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            issues.append(DatasetIssue(None, 'database_unavailable', str(error)))
    if not checked:
        if not any(i.code == 'database_unavailable' for i in issues):
            issues.append(DatasetIssue(None, 'database_unavailable', 'Chunk existence cannot be validated'))
    else:
        # The legacy validator also returns a metadata-updated COPY. Discard it;
        # the frozen Golden and its chunk IDs are never rewritten.
        evidence, _ = validate_evidence(records, chunks, paper_ids)
        for paper_id in evidence['missing_paper_ids']:
            for row in records:
                if row.get('paper_id') == paper_id:
                    issues.append(DatasetIssue(row.get('id'), 'invalid_paper_id', str(paper_id)))
        for key in ('missing_chunk_ids', 'paper_id_mismatches', 'bibliography_classification_errors',
                    'section_type_mismatches'):
            issues.extend(DatasetIssue(e.get('case_id'), key, str(e)) for e in evidence[key])
    return GoldenValidation(len(records), tuple(dict.fromkeys(issues)),
        tuple(sorted({k for row in records for k in row if not k.startswith('__')})), checked)


def load_validated_golden(path: Path, database: Path):
    rows, errors = load_jsonl(path)
    return rows, validate_golden_dataset(rows, database=database, parse_errors=errors)
