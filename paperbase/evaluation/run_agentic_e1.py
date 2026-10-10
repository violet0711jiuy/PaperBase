"""Opt-in E1 run/replay. Replay never constructs an Agent or provider client."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from .contracts import CaseTrace, EVALUATION_SCHEMA_VERSION, RunManifest, json_safe
from .dataset import load_validated_golden
from .diagnostic import evaluate_agentic_case
from .reporting import write_evaluation_report
from .trace import trace_from_artifacts, trace_from_result, result_artifact
from .reclassify_agentic_eval import _read_json
from paperbase.llm.answer_output_capture import capture_answer_outputs

ROOT = Path(__file__).resolve().parents[2]
SMOKE_CASE_IDS = ('fact_001','method_001','method_002','experiment_001','result_001',
                  'synthesis_001','bibliography_001','unanswerable_001')


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_snapshot(root=ROOT):
    """Git alone cannot identify this dirty workspace; include content tree hash."""
    tree=hashlib.sha256()
    for path in sorted((root/'paperbase').rglob('*.py')):
        tree.update(str(path.relative_to(root)).encode());tree.update(path.read_bytes())
    try:
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()
    except (OSError,subprocess.CalledProcessError):
        commit='unknown'
    return f'{commit}:paperbase-sha256:{tree.hexdigest()}'


def safe_model_summary(config):
    """An allowlist: keys, secrets, prompts and hidden reasoning are never serialized."""
    allowed=('provider','model','temperature','structured_output_mode','timeout_seconds','max_tokens')
    summary={k:config[k] for k in allowed if k in config and isinstance(config[k],(str,int,float,bool,type(None)))}
    if isinstance(config.get('base_url'),str):
        from urllib.parse import urlsplit
        summary['endpoint_host']=urlsplit(config['base_url']).hostname
    return summary


def replay(*, dataset: Path, source: Path, output: Path, database: Path,
           case_ids=(), smoke=False):
    from .run_agentic_eval import _write_json
    source=source.resolve();output=output.resolve();dataset=dataset.resolve()
    if output==source or output.is_relative_to(source):
        raise ValueError('Replay output must be outside the historical source tree')
    rows,validation=load_validated_golden(dataset,database)
    requested=set(SMOKE_CASE_IDS if smoke else case_ids)
    if requested-set(r.get('id') for r in rows):
        raise ValueError('Unknown case IDs in frozen dataset')
    rows=[r for r in rows if not requested or r.get('id') in requested]
    old_rows=[]
    if (source/'case_results.jsonl').exists():
        old_rows=[json.loads(l) for l in (source/'case_results.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    old_by_id={str(r.get('id')):r for r in old_rows}
    records=[];hashes=[]
    if (source/'case_results.jsonl').exists():
        hashes.append((str(source/'case_results.jsonl'),_sha(source/'case_results.jsonl')))
    for row in rows:
        case_id=str(row.get('id','unknown'))
        # Invalid IDs are dataset errors; never use them as filesystem paths.
        safe_id=case_id if re.fullmatch(r'[A-Za-z0-9_-]+',case_id) else 'invalid_'+hashlib.sha256(case_id.encode()).hexdigest()[:12]
        result_path=source/'cases'/f'{safe_id}.result.json';audit_path=source/'cases'/f'{safe_id}.audit.json'
        artifacts=[];artifact_gaps=[]
        for path in (result_path,audit_path):
            try:
                artifacts.append(_read_json(path));hashes.append((str(path),_sha(path)))
            except (OSError,ValueError):
                artifacts.append({});artifact_gaps.append(f'missing_or_invalid_{path.suffixes[-2][1:]}_artifact')
        old=old_by_id.get(case_id,{})
        # Top-5 is comparable only when the old row explicitly identifies the
        # same chunk-level gold. Do not treat an initial Agent search as v1's
        # separate standalone Retriever observation.
        from .agentic import gold_chunk_ids
        ranked=old.get('retrieved_chunk_ids') if set(old.get('gold_chunk_ids',()))==set(gold_chunk_ids(row)) else None
        try:
            trace=trace_from_artifacts(*artifacts,legacy_ranked_chunk_ids=ranked)
        except (ValueError,TypeError,KeyError):
            trace=CaseTrace(None,None,None,None,None,None,None,None,
                observability_gaps=('invalid_saved_case_trace',))
        if artifact_gaps:
            from dataclasses import replace
            trace=replace(trace,observability_gaps=(*trace.observability_gaps,*artifact_gaps))
        issues=tuple(i for i in validation.issues if i.case_id in (None,case_id))
        record=evaluate_agentic_case(row,trace,dataset_issues=issues);records.append(record)
        _write_json(output/'cases'/f'{safe_id}.evaluation.json',{**record.to_dict(),
            'source_artifacts':dict(result_path=str(result_path),audit_path=str(audit_path))})
    manifest=RunManifest(output.name,EVALUATION_SCHEMA_VERSION,str(dataset),_sha(dataset),len(records),
        datetime.now(timezone.utc).isoformat(),'historical_replay_smoke' if smoke else 'historical_replay',
        {'model':None,'provider':None,'source':'historical config unavailable; not current model config'},
        {'source':'historical budget/settings unavailable; observed counters are in each evaluation'},None,
        code_snapshot(),source_artifact_hashes=tuple(hashes),observability_gaps=(
            'historical_model_provider_config','historical_agent_config','historical_semantic_authority_version'))
    _write_json(output/'golden_validation.json',{**json_safe(validation),'valid':validation.valid})
    _write_json(output/'smoke_set.json',dict(case_ids=list(SMOKE_CASE_IDS),live_provider_executed=False))
    summary=write_evaluation_report(output,records,manifest,old_rows=old_rows)
    for path,digest in hashes:
        assert _sha(Path(path))==digest,'Historical input was modified'
    return records,summary


def live(*, dataset: Path, output: Path, database: Path, config: Path | None = None,
         case_ids=(), smoke=False):
    """Explicit CLI action only. Never used by pytest or offline replay."""
    from .run_agentic_eval import _write_json
    rows,validation=load_validated_golden(dataset,database)
    requested=set(SMOKE_CASE_IDS if smoke else case_ids)
    if requested-set(r.get('id') for r in rows):
        raise ValueError('Unknown case IDs in frozen dataset')
    rows=[r for r in rows if not requested or r.get('id') in requested]
    # Validate the entire frozen dataset before creating any real model client.
    if not validation.valid:
        records=[evaluate_agentic_case(row,CaseTrace(None,None,None,None,None,None,None,None),
            dataset_issues=tuple(i for i in validation.issues if i.case_id in (None,row.get('id')))) for row in rows]
        manifest=RunManifest(output.name,EVALUATION_SCHEMA_VERSION,str(dataset.resolve()),_sha(dataset),len(rows),
            datetime.now(timezone.utc).isoformat(),'dataset_validation_failed',{}, {},None,code_snapshot())
        _write_json(output/'golden_validation.json',{**json_safe(validation),'valid':False})
        write_evaluation_report(output,records,manifest)
        return records
    # Reuse the existing runner's normal production factory and safe public audit.
    from paperbase.agentic.factory import create_agentic_research_service
    from paperbase.agentic.audit import build_agentic_audit_snapshot
    from paperbase.agentic.graph import _SEMANTIC_AUTHORITY_VERSION
    from paperbase.config import load_settings
    from paperbase.llm.client import load_llm_runtime_settings
    from langgraph.checkpoint.memory import InMemorySaver
    settings=load_settings(config)
    runtime=load_llm_runtime_settings(settings.config_path.parent/'.env')
    model=safe_model_summary(vars(runtime))
    records=[]
    manifest=RunManifest(output.name,EVALUATION_SCHEMA_VERSION,str(dataset.resolve()),_sha(dataset),len(rows),
        datetime.now(timezone.utc).isoformat(),'live_smoke' if smoke else 'live',model,
        settings.agentic_research.model_dump(mode='json'),_SEMANTIC_AUTHORITY_VERSION,code_snapshot())
    _write_json(output/'run_manifest.json',json_safe(manifest))
    _write_json(output/'golden_validation.json',{**json_safe(validation),'valid':True})
    try:
        service=create_agentic_research_service(config_path=settings.config_path,checkpointer=InMemorySaver())
    except Exception as error:
        from .trace import TRACE_SCHEMA_VERSION
        for row in rows:
            case_id=str(row['id'])
            trace=CaseTrace('failed','fatal_error',None,None,None,None,None,None,
                setup_failure_stage='scope',observability_gaps=(f'service_setup_exception:{type(error).__name__}',))
            record=evaluate_agentic_case(row,trace);records.append(record)
            _write_json(output/'cases'/f'{case_id}.result.json',dict(run_status='failed',stop_reason='fatal_error',
                error_code=type(error).__name__,case_trace_schema_version=TRACE_SCHEMA_VERSION,case_trace=json_safe(trace)))
            _write_json(output/'cases'/f'{case_id}.audit.json',{})
            _write_json(output/'cases'/f'{case_id}.evaluation.json',record.to_dict())
        write_evaluation_report(output,records,manifest)
        return records
    try:
        for row in rows:
            case_id=str(row['id'])
            if not re.fullmatch(r'[A-Za-z0-9_-]+',case_id):
                raise ValueError('Case ID cannot be used as artifact filename')
            try:
                with capture_answer_outputs(output/'cases'/f'{case_id}.answer_llm_outputs.jsonl', case_id=case_id):
                    result=service.research(row['question'],selected_paper_ids=(row['paper_id'],),thread_id=f'e1-{output.name}-{case_id}')
            except Exception as error:
                # Safe exception class only, no provider response/API credential.
                trace=CaseTrace('failed','fatal_error',None,None,None,None,None,None,
                    setup_failure_stage='scope',observability_gaps=(f'run_setup_exception:{type(error).__name__}',))
                from .trace import TRACE_SCHEMA_VERSION
                _write_json(output/'cases'/f'{case_id}.result.json',dict(run_status='failed',stop_reason='fatal_error',
                    case_trace_schema_version=TRACE_SCHEMA_VERSION,case_trace=json_safe(trace),error_code=type(error).__name__))
                _write_json(output/'cases'/f'{case_id}.audit.json',{})
            else:
                # Serialization/evaluator errors are not relabeled Agent errors.
                trace=trace_from_result(result)
                _write_json(output/'cases'/f'{case_id}.result.json',result_artifact(result))
                _write_json(output/'cases'/f'{case_id}.audit.json',build_agentic_audit_snapshot(result).to_dict())
            record=evaluate_agentic_case(row,trace);records.append(record)
            _write_json(output/'cases'/f'{case_id}.evaluation.json',record.to_dict())
    finally:
        service.close()
    write_evaluation_report(output,records,manifest)
    return records


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--source',type=Path,help='Saved result/audit directory: offline, zero provider calls')
    mode.add_argument('--live',action='store_true',help='Explicitly authorize real provider use')
    parser.add_argument('--dataset',type=Path,default=ROOT/'eval/datasets/golden_dataset_v1_2.jsonl')
    parser.add_argument('--database',type=Path,default=ROOT/'storage/paperbase.sqlite3')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--case-id',action='append',default=[])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args(argv)
    if args.smoke and args.case_id:
        parser.error('--smoke and --case-id are separate selections')
    common=dict(dataset=args.dataset,output=args.output,database=args.database,case_ids=args.case_id,smoke=args.smoke)
    replay(source=args.source,**common) if args.source is not None else live(config=args.config,**common)
    print(args.output.resolve())
    return 0


if __name__=='__main__':
    raise SystemExit(main())
