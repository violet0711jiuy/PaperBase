"""Aggregate typed E1 records with explicit known-case denominators."""
from __future__ import annotations

from collections import Counter
from dataclasses import fields
import hashlib
import json
from pathlib import Path

from .contracts import EVALUATION_SCHEMA_VERSION, AgenticCaseEvaluation, json_safe


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return dict(value=sum(values)/len(values) if values else None, observed_cases=len(values))


def aggregate_evaluations(records: list[AgenticCaseEvaluation]) -> dict:
    valid = [r for r in records if r.primary_failure_stage != 'dataset']
    summary = dict(evaluation_schema_version=EVALUATION_SCHEMA_VERSION,case_count=len(records),
        dataset_error_cases=len(records)-len(valid),agent_case_count=len(valid),
        primary_failure_counts=dict(Counter(r.primary_failure_stage or 'none' for r in valid)),
        failure_label_counts=dict(Counter(l for r in valid for l in r.failure_labels)),
        observability_gap_cases=sum(r.observability_gap for r in valid),
        observability_gaps=dict(Counter(g for r in valid for g in r.observability_gaps)),
        stop_reasons=dict(Counter(r.stop_reason or 'unknown' for r in valid)))
    for group in ('retrieval','efficiency','answer'):
        dto = getattr(valid[0],group) if valid else None
        summary[group] = {f.name:_mean(getattr(getattr(r,group),f.name) for r in valid)
            for f in fields(dto) if f.name not in {'invalid_citation_ids','status'}} if dto else {}
    summary['research'] = {name:_mean(getattr(r.research,name) for r in valid) for name in
        ('planned_requirement_count','required_requirement_count','answered_requirement_count',
         'open_gap_requirement_count','requirement_answer_rate')}
    funnel={}
    for stage in ('retrieved','registered','answer_input','cited'):
        applicable=[r for r in valid if r.answerable and r.evidence_funnel.gold_chunk_ids
            and getattr(r.evidence_funnel,f'gold_{stage}_chunk_ids') is not None]
        denominator=sum(len(r.evidence_funnel.gold_chunk_ids) for r in applicable)
        numerator=sum(len(getattr(r.evidence_funnel,f'gold_{stage}_chunk_ids')) for r in applicable)
        funnel[f'gold_{stage}_rate']=dict(value=numerator/denominator if denominator else None,
            gold_chunks_observed=denominator,gold_chunks_at_stage=numerator,observed_cases=len(applicable))
        path_field = f'path_covered_gold_{stage}_chunk_ids'
        path_applicable = [
            r for r in valid if r.answerable and r.evidence_funnel.gold_chunk_ids
            and getattr(r.evidence_funnel, path_field) is not None
        ]
        path_denominator = sum(
            len(r.evidence_funnel.gold_chunk_ids) for r in path_applicable
        )
        path_numerator = sum(
            len(getattr(r.evidence_funnel, path_field)) for r in path_applicable
        )
        funnel[f'evidence_path_{stage}_rate'] = dict(
            value=path_numerator/path_denominator if path_denominator else None,
            gold_chunks_observed=path_denominator,
            gold_chunks_at_stage=path_numerator,
            observed_cases=len(path_applicable),
        )
    for before,after in [('retrieved','registered'),('registered','answer_input'),('answer_input','cited')]:
        after_field = f'gold_{after}_chunk_ids'
        applicable=[r for r in valid if r.answerable and r.evidence_funnel.gold_chunk_ids
            and getattr(r.evidence_funnel,f'gold_{before}_chunk_ids') is not None
            and getattr(r.evidence_funnel,after_field) is not None]
        pairs=[(set(getattr(r.evidence_funnel,f'gold_{before}_chunk_ids')),
                set(getattr(r.evidence_funnel,after_field))) for r in applicable]
        denominator=sum(len(a) for a,b in pairs);numerator=sum(len(a & b) for a,b in pairs)
        funnel[f'{before}_to_{after}']=dict(value=numerator/denominator if denominator else None,
            eligible_chunks=denominator,converted_chunks=numerator,observed_cases=len(applicable))
    summary['evidence_funnel']=funnel
    return summary


def metrics_by_primary_type(records):
    return {kind:aggregate_evaluations([r for r in records if r.primary_type==kind])
        for kind in sorted({r.primary_type for r in records})}


def historical_comparison(records, old_rows):
    """Compare paired unchanged definitions, never rebrand missing v1 Funnel as zero."""
    old={str(row.get('id')):row for row in old_rows}
    pairs=[]
    for record in records:
        before=old.get(record.case_id)
        if before is None:
            continue
        same_gold=before.get('gold_chunk_ids') is not None and set(before['gold_chunk_ids'])==set(record.evidence_funnel.gold_chunk_ids)
        pairs.append(dict(case_id=record.case_id,tool_calls=dict(old=before.get('tool_calls'),new=record.efficiency.tool_calls),
            retrieval_queries=dict(old=before.get('retrieval_queries'),new=record.efficiency.retrieval_queries),
            stop_reason=dict(old=before.get('stop_reason'),new=record.stop_reason),
            hit_at_5=dict(old=before.get('hit_at_5'),new=record.retrieval.hit_at_5) if same_gold else None,
            recall_at_5=dict(old=before.get('recall_at_5'),new=record.retrieval.recall_at_5) if same_gold else None))
    return dict(baseline_role='historical_baseline',paired_cases=pairs,
        definitions='Old evaluator != eval-v2. Legacy standalone Top-5 only; initial/cumulative metrics differ.',
        old_funnel=None,failure_counts_directly_comparable=False)


def render_report(summary, by_type, records, manifest):
    def display(v):
        return 'null' if v is None else f'{v:.3f}' if isinstance(v,(float,int)) else str(v)
    sections=[f'# E1 Diagnostic Evaluation — {manifest.run_id}',
        f'模式：{manifest.mode}；schema：{manifest.evaluation_schema_version}；case 数：{summary["case_count"]}。',
        '这是确定性诊断，不报告开放式语义正确率。Requirement answer 与 source-span citation 是同一语义边界。',
        '## Retrieval', '| Metric | Value | Known cases |','|---|---:|---:|']
    sections.extend(f'| {name} | {display(row["value"])} | {row["observed_cases"]} |' for name,row in summary['retrieval'].items())
    sections+=['## Evidence funnel','| Stage / conversion | Value | Known cases |','|---|---:|---:|']
    sections.extend(f'| {name} | {display(row["value"])} | {row["observed_cases"]} |' for name,row in summary['evidence_funnel'].items())
    for title,key in [('Requirement research','research'),('Agent efficiency','efficiency'),('Answer safety','answer')]:
        sections += [f'## {title}','```json',json.dumps(summary[key],ensure_ascii=False,indent=2),'```']
    sections+=['## Failure taxonomy','```json',json.dumps({k:summary[k] for k in ('dataset_error_cases','primary_failure_counts','failure_label_counts','stop_reasons')},ensure_ascii=False,indent=2),'```',
        '## Metrics by primary_type','| Type | Cases | Primary stages | Gap cases |','|---|---:|---|---:|']
    sections.extend(f'| {kind} | {m["case_count"]} | {m["primary_failure_counts"]} | {m["observability_gap_cases"]} |' for kind,m in by_type.items())
    sections+=['## Representative failures']
    sections.extend(f'- {r.case_id}: primary={r.primary_failure_stage}; labels={list(r.failure_labels)}; stop={r.stop_reason}; gold retrieved/registered/answer-input/cited={r.evidence_funnel.gold_retrieved_chunk_ids}/{r.evidence_funnel.gold_registered_chunk_ids}/{r.evidence_funnel.gold_answer_input_chunk_ids}/{r.evidence_funnel.gold_cited_chunk_ids}'
        for r in [r for r in records if r.failure_labels][:8])
    sections+=['## Observability gaps','```json',json.dumps(summary['observability_gaps'],ensure_ascii=False,indent=2),'```',
        '未知不是零；null 指标不进入均值，Funnel 使用已观测 gold chunk 加权分母。',
        '旧 failure count 与 v2 定义不同，不直接比较。历史输出不能预设当前冻结系统的失败。',
        '详细定义见 docs/evaluation/e1_diagnostic.md。E2 Judge 未调用。']
    return '\n\n'.join(sections)+'\n'


def write_evaluation_report(output: Path, records, manifest, *, old_rows=()):
    from .run_agentic_eval import _write_json
    output.mkdir(parents=True,exist_ok=True)
    summary=aggregate_evaluations(records);by_type=metrics_by_primary_type(records)
    _write_json(output/'summary.json',summary)
    _write_json(output/'metrics_by_type.json',by_type)
    _write_json(output/'failure_cases.json',[r.to_dict() for r in records if r.failure_labels])
    _write_json(output/'run_manifest.json',json_safe(manifest))
    _write_json(output/'historical_baseline_comparison.json',historical_comparison(records,old_rows))
    (output/'evaluation_report.md').write_text(render_report(summary,by_type,records,manifest),encoding='utf-8')
    return summary
