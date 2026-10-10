"""Generate a read-only, zero-provider PaperBase architecture snapshot.

Only this tooling and the designated snapshot/output files are written. Curated
explanations are tied to the reviewed tree; changed production source requires
manual architecture review, even when deterministic assembly still succeeds.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from contextlib import ExitStack
from dataclasses import fields, is_dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCHEMA = "paperbase-architecture-snapshot-v1"
CURATED_TREE = "54766b03327438e05f56f3d7a6777907c5c089541c9eb1aba8463905c02f5e7d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def bundle(rows):
    return hashlib.sha256("".join(f"{p}\t{v}\n" for p, v in sorted(rows.items())).encode()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_of(obj):
    return dict(path=Path(inspect.getsourcefile(obj)).relative_to(ROOT).as_posix(),
                line=inspect.getsourcelines(obj)[1])


def source_link(location, label=None):
    return f'[{label or location["path"]}](../../../{location["path"]}#L{location["line"]})'


def protected_files():
    paths = [p for base in ("paperbase", "app", "tests", "eval", "docs") for p in (ROOT / base).rglob("*")
             if p.is_file() and p.suffix in (".py", ".md", ".json", ".jsonl", ".yaml")
             and "current_snapshot" not in p.parts and "results" not in p.parts]
    paths += list(ROOT.glob("config*.yaml")) + [ROOT / ".env", ROOT / ".env.example", ROOT / "README.md"]
    paths += [p for p in (ROOT / "storage").glob("*") if p.is_file() and ".sqlite" in p.name]
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if p.exists()}


def inspect_runtime():
    from paperbase.config import load_settings
    from paperbase.llm.client import create_chat_completion_client, load_llm_runtime_settings
    from paperbase.agentic.factory import create_agentic_research_service
    from paperbase.agentic.contracts import (
        QueryContext, QueryScope, ScopeSource, Intent, ResearchGoal, ResearchRequirement,
        ResearchPlan, EvidenceTarget, EvidenceProjectionType, BudgetUsage, RegistryState,
    )
    from paperbase.agentic.decision import DecisionState, _decision_json_schema
    from paperbase.agentic.checkpoint_serde import dump_contract, load_contract
    from paperbase.agentic.schemas import QueryUnderstandingDraft, CandidateSelectionDraft, ResearchPlanDraft, transport_projection_model
    from paperbase.agentic.evidence_projection import _projection_schema
    from paperbase.generation.answer_generator import _answer_response_schema
    from paperbase.evaluation.dataset import load_validated_golden
    from langgraph.checkpoint.memory import InMemorySaver
    from openai._constants import DEFAULT_MAX_RETRIES

    settings = load_settings(ROOT / "config.yaml")
    # Isolate this subprocess from pre-existing process-level provider overrides.
    with patch.dict(os.environ, {k:v for k,v in os.environ.items() if not k.startswith(("LLM_", "ZHIPUAI_"))}, clear=True):
        runtime = load_llm_runtime_settings(ROOT / ".env")
    requests = []
    network_attempts = []
    checks = []

    def forbid_network(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError("Architecture inspection must not access network/provider")

    class SDK:
        def __init__(self, **kwargs):
            self.max_retries = DEFAULT_MAX_RETRIES
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            # Only effective request metadata retained, never messages/content.
            fmt = kwargs.get("response_format", {})
            schema = fmt.get("json_schema", {})
            requests.append(dict(model=kwargs["model"], temperature=kwargs["temperature"],
                output_mode=fmt.get("type"), strict=schema.get("strict", False),
                schema_name=schema.get("name"), schema_sha256=json_hash(schema.get("schema")) if schema else None,
                top_p=kwargs.get("top_p"), presence_penalty=kwargs.get("presence_penalty"),
                extra_body=kwargs.get("extra_body", {}), timeout=kwargs.get("timeout"),
                max_tokens=kwargs.get("max_tokens"), max_tokens_key_present="max_tokens" in kwargs))
            return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(
                content="{}", reasoning_content=None))], usage=SimpleNamespace(prompt_tokens=0, completion_tokens=0))

    class Database:
        def initialize(self): pass
        def list_documents(self): return ()
        def get_document(self, paper_id): return None

    class Retriever:
        pass

    with ExitStack() as stack:
        stack.enter_context(patch.object(socket, "create_connection", forbid_network))
        stack.enter_context(patch.object(socket.socket, "connect", forbid_network))
        stack.enter_context(patch("openai.OpenAI", SDK))
        imported, cli = [], []
        for base in ("paperbase", "app"):
            for p in sorted((ROOT / base).rglob("*.py")):
                parts = p.relative_to(ROOT).with_suffix("").parts
                name = ".".join(parts[:-1] if parts[-1] == "__init__" else parts)
                if parts[-1] == "__main__":
                    compile(p.read_text(encoding="utf-8-sig"), str(p), "exec")
                    cli.append(name)
                else:
                    importlib.import_module(name)
                    imported.append(name)
        checks.append(dict(name="production_library_imports", passed=True, count=len(imported),
            cli_entrypoints_compile_only=cli, explanation="__main__ wrappers intentionally execute CLI; underlying implementation modules imported, no CLI launched."))
        base = create_chat_completion_client(runtime)
        base.complete_json_diagnostic(system_prompt="Offline metadata inspection", user_prompt="{}",
            json_schema=QueryUnderstandingDraft.model_json_schema(), schema_name="snapshot_base_probe")
        base_request = requests[-1]
        checks.append(dict(name="base_client_constructed", passed=True, sdk="injected no-network sentinel; actual production client constructor"))
        context = QueryContext("architecture fixture", "architecture fixture", Intent.RETRIEVE,
            QueryScope.SINGLE_PAPER, ("SNAPSHOT_PAPER",), ScopeSource.UI_SELECTION, ResearchGoal.FIND_EVIDENCE,
            frozenset({EvidenceTarget.GENERAL}))
        plan = ResearchPlan("architecture fixture", (ResearchRequirement("R1", "architecture fixture",
            frozenset({EvidenceTarget.GENERAL}), projection_type=EvidenceProjectionType.FACT),))
        state = DecisionState(context, plan, (), (), BudgetUsage())
        schemas = dict(query_understanding=QueryUnderstandingDraft.model_json_schema(),
            candidate_selection=CandidateSelectionDraft.model_json_schema(), planner=ResearchPlanDraft.model_json_schema(),
            decision=_decision_json_schema(state),
            projector={p.value:dict(batch=transport_projection_model(p.value).model_json_schema(),
                                   single=_projection_schema(p)) for p in EvidenceProjectionType},
            answer=_answer_response_schema((("R1", "architecture fixture"),), {"R1":("E1",)}),
            classic_answer=_answer_response_schema((), {}, ("E1",)))
        assert load_contract(dump_contract(RegistryState(), RegistryState), RegistryState) == RegistryState()
        checks += [dict(name="checkpoint_serde_roundtrip", passed=True), dict(name="schemas_built", passed=True,
                   schema_hashes={k:json_hash(v) for k,v in schemas.items()}, dynamic_scope="synthetic IDs only; no user State/Evidence" )]
        with TemporaryDirectory(prefix="paperbase-snapshot-") as temp:
            isolated = settings.model_copy(update={"agentic_research":settings.agentic_research.model_copy(update={
                "evidence_analysis_cache_path":Path(temp)/"cache.sqlite3"})})
            with patch("paperbase.agentic.factory.load_settings", return_value=isolated):
                service = create_agentic_research_service(config_path=ROOT/"config.yaml", client=base,
                    database=Database(), hybrid_retriever=Retriever(), checkpointer=InMemorySaver())
            ctx = service._workflow._context
            graph = service._workflow.graph.get_graph()
            graph_meta = dict(nodes=list(graph.nodes), edges=[dict(source=e.source,target=e.target,conditional=e.conditional) for e in graph.edges])
            node_clients = dict(query_understanding=ctx.query_understanding._client,
                candidate_selection=ctx.candidate_selector._client, planner=ctx.planner._client,
                decision=ctx.decision_agent._client, projector=ctx.evidence_analyzer._client,
                answer=ctx.answer_generator._client)
            sdk_names = dict(query_understanding="paperbase_query_understanding", candidate_selection="paperbase_candidate_selection",
                planner="paperbase_research_plan", decision="paperbase_agent_action",
                projector="paperbase_evidence_transport_batch", answer="paperbase_grounded_answer_semantic_v2")
            node_requests = {}
            for name, client in node_clients.items():
                client.complete_json_diagnostic(system_prompt="Offline metadata inspection", user_prompt="{}",
                    json_schema=schemas["projector"]["fact"]["batch"] if name=="projector" else schemas[name], schema_name=sdk_names[name])
                node_requests[name] = requests[-1]
            checks.append(dict(name="actual_agentic_request_assembly", passed=all(
                not req['max_tokens_key_present'] and
                req['output_mode']==('json_object' if name in ('query_understanding','candidate_selection') else 'json_schema') and
                req['strict']==(name not in ('query_understanding','candidate_selection')) and
                req['temperature']==(1.0 if name=='decision' else 0.7) and
                req['extra_body'].get('enable_thinking')==(name=='decision')
                for name,req in node_requests.items()),
                method="Actual production client views + native complete_json_diagnostic create metadata, injected mock SDK only."))
            assert service._workflow.graph is not None
            checks.append(dict(name="production_factory_service_graph", passed=True,
                dependencies="Injected SDK sentinel, empty DB/retriever, in-memory checkpointer, temporary cache; no production storage mutation or local model inference."))
            actual_internal = dict(planner_max_attempts=ctx.planner._max_attempts,
                decision_max_attempts=ctx.decision_agent._max_attempts, projector_max_attempts=ctx.evidence_analyzer._max_attempts,
                candidate_max_attempts=ctx.candidate_selector._max_attempts, candidate_batch_size=ctx.candidate_selector._batch_size)
            service.close()
        rows, golden_validation = load_validated_golden(ROOT/"eval/datasets/golden_dataset_v1_2.jsonl", settings.database.path)
        assert golden_validation.valid
        taxonomy = dict(sorted(Counter(r["primary_type"] for r in rows).items()))
        checks.append(dict(name="frozen_golden_read_only_validation", passed=True, cases=len(rows), taxonomy=taxonomy))
    assert not network_attempts
    model = dict(provider=runtime.provider, model=runtime.model, endpoint_host=urlsplit(runtime.base_url).hostname,
        sampling_profile=settings.agentic_research.llm_sampling_profile,
        base_client=base_request, sdk_default_max_retries=DEFAULT_MAX_RETRIES,
        sdk_version=importlib.metadata.version("openai"), credential_configured=bool(runtime.api_key))
    return dict(settings=settings, runtime_secret=runtime.api_key, model=model, node_requests=node_requests,
                graph=graph_meta, schemas=schemas, taxonomy=taxonomy, golden_count=len(rows),
                checks=checks, imported_modules=imported, actual_internal=actual_internal, mock_sdk_creates=len(requests),
                network_attempts=len(network_attempts), provider_calls=0)


CONTRACT_DESCRIPTIONS = {
    "QueryContext":("Scope / resolved query / advisory hints", "ScopeResolver + QueryUnderstandingService", "Planner, Decision, Tools", "public domain", True),
    "ResearchPlan":("Ordered answering obligations", "ResearchPlanner", "Graph, Decision, Projector, Selector, Answer", "public domain", True),
    "ResearchRequirement":("Requirement ID / semantics / required / target / projection shape", "ResearchPlanner", "Search bindings, Projector, Integrity, Answer", "public domain (nested Plan)", True),
    "AgentAction":("Typed chosen tool command", "LLMDecisionAgent after local validation", "Graph, Guard, Tool executor", "public domain union", True),
    "CanonicalSearchRequest":("Compiler-normalized effective retrieval input and identity", "retrieval.requests compilers / compile_action_requests", "Guard, Executor, Retriever, receipts", "internal domain union", True),
    "RuntimeActionCapabilities":("Structural feasible domain derived from current Graph State", "decision._runtime_action_capabilities", "Decision View / dynamic Schema / local validation", "internal derived view + compatibility DTO", False),
    "ActionRejectionContext":("Current/historical rejection feedback, not replacement Action", "Decision preflight / Graph runtime validation", "Decision View, audit, checkpoint", "public domain", True),
    "RetrievedEvidence":("Retrieval content locator / provenance / explicit score contract", "Tools and retrieval adapters", "Admission, Registry", "public tool/domain DTO", True),
    "EvidenceRecord":("Run-local Evidence identity, binding, successful assessment", "EvidenceRegistry", "Projector, Integrity, State, Selector, Answer, audit", "public domain", True),
    "EvidenceProjection":("Typed local structured extraction, not final completeness", "LLMEvidenceProjector", "Integrity, Registry, Selector, Answer", "public domain union nested in assessment", True),
    "EvidenceAnalysis":("Per-Evidence assessment aggregate and explicit evaluated scope", "Projector + merge_analysis_groups", "Registry / checkpoint / diagnostics", "public domain", True),
    "ProjectionSemanticRequest":("Question + scoped Requirement meaning used in semantic cache identity", "build_projection_semantic_request", "Projector prompt scope / cache fingerprint", "internal ephemeral", False),
    "RequirementState":("Research attempt / local verified support status", "track/derive/finalize_requirement_states", "Decision, Graph Finish closure, audit", "public domain", True),
    "CandidateSelection":("Frozen corpus candidate / uncertain / excluded disposition", "LLMCandidateSelector (metadata-only) or explicit-scope candidate derivation", "Coverage, Guard, Decision", "public domain", True),
    "AnswerSemanticDraft":("Untrusted provider text + supported-required semantic status", "Answer LLM", "Local JSON/schema/domain parser and deterministic compiler", "internal provider draft", False),
    "CompletionView":("Selected verified support map for required IDs", "build_completion_view", "Answer Schema / allowed coverage", "internal ephemeral", False),
    "AnswerRequirementCoverage":("Required Requirement status and actually cited allowed support", "compile_answer_requirement_coverage", "derive_overall_answer_status, ResearchAnswer, E1", "public final DTO nested in ResearchAnswer", True),
    "ResearchAnswer":("Final user-facing typed result", "generate_agentic_answer / Graph deterministic short circuits", "Service / UI / audit / E1", "public final DTO", True),
}


def catalog():
    from paperbase.agentic import contracts as c
    from paperbase.retrieval import requests as r, contracts as rc
    from paperbase.agentic.projection_semantics import ProjectionSemanticRequest
    from paperbase.agentic.answering import CompletionView
    from paperbase.generation.answer_generator import AnswerSemanticDraft
    from paperbase.generation.answer_coverage import AnswerRequirementCoverage
    special = dict(CanonicalSearchRequest=r.CanonicalSearchRequest, RetrievedEvidence=rc.RetrievedEvidence,
        ProjectionSemanticRequest=ProjectionSemanticRequest, CompletionView=CompletionView,
        AnswerSemanticDraft=AnswerSemanticDraft, AnswerRequirementCoverage=AnswerRequirementCoverage)
    result={}
    for name,(purpose,producer,consumer,visibility,checkpoint) in CONTRACT_DESCRIPTIONS.items():
        obj=special[name] if name in special else getattr(c,name)
        union=get_args(obj)
        models=union if union else (obj,)
        variants=[]
        for model in models:
            location=source_of(model)
            tree=ast.parse((ROOT/location['path']).read_text(encoding='utf-8-sig'))
            cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==model.__name__)
            annotations={n.target.id:ast.unparse(n.annotation) for n in cls.body
                         if isinstance(n,ast.AnnAssign) and isinstance(n.target,ast.Name)}
            field_names=list(model.model_fields) if hasattr(model,'model_fields') else [f.name for f in fields(model)]
            variants.append(dict(name=model.__name__,source=location,core_fields=[dict(name=f,type=annotations.get(f,'inherited')) for f in field_names]))
        result[name]=dict(purpose=purpose,producer=producer,consumers=consumer,visibility=visibility,
            checkpoint=checkpoint,serialization='Pydantic JSON validation then domain DTO; nested through checkpoint_serde' if checkpoint else 'derived JSON/view or local provider validation; not stored as provider draft',
            variants=variants)
    return result


def versions_and_authorities():
    from paperbase.agentic import graph, grounding, guards, decision, search_requests, selector, answering, schemas
    from paperbase.generation import answer_coverage, answer_generator
    from paperbase.agentic import projection_semantics, evidence_analysis_cache, checkpoint_serde
    from paperbase.prompts import agentic_query_understanding as qu, agentic_planner as pl, agentic_decision as de, evidence_projector as pr, answer_generation as an, candidate_selection as ca
    from paperbase.evaluation.contracts import EVALUATION_SCHEMA_VERSION
    modules=dict(query_understanding=qu,planner=pl,decision=de,projector=pr,answer=an,candidate_selection=ca)
    texts=dict(query_understanding=qu.QUERY_UNDERSTANDING_SYSTEM_PROMPT,planner=pl.RESEARCH_PLANNER_SYSTEM_PROMPT,
        decision=de.AGENT_DECISION_SYSTEM_PROMPT,projector=pr.EVIDENCE_PROJECTOR_SYSTEM_PROMPT,
        answer=an.ANSWER_SEMANTIC_SYSTEM_PROMPT,candidate_selection=ca.CANDIDATE_SELECTION_SYSTEM_PROMPT)
    prompts={name:dict(version=getattr(module,{'projector':'EVIDENCE_PROJECTOR_PROMPT_VERSION','answer':'ANSWER_SEMANTIC_PROMPT_VERSION'}.get(name,'_NO_VERSION'),'UNVERSIONED'),
        source=Path(module.__file__).relative_to(ROOT).as_posix(),source_sha256=sha(Path(module.__file__)),
        effective_text_sha256=hashlib.sha256(texts[name].encode()).hexdigest()) for name,module in modules.items()}
    schema_versions={name:dict(version='UNVERSIONED',schema_names=[schema_name]) for name,schema_name in (
        ('query_understanding','paperbase_query_understanding'),('candidate_selection','paperbase_candidate_selection'),
        ('planner','paperbase_research_plan'),('decision','paperbase_agent_action'))}
    schema_versions['projector']=dict(version=evidence_analysis_cache.EVIDENCE_ANALYSIS_SCHEMA_VERSION,
        schema_names=['paperbase_evidence_transport_batch','paperbase_evidence_projection'])
    schema_versions['answer']=dict(version=answer_coverage.ANSWER_SEMANTIC_SCHEMA_VERSION,
        schema_names=['paperbase_grounded_answer_semantic_v2'],classic_schema_names=['paperbase_grounded_answer'])
    authority=dict(search_request='retrieval.requests canonical compiler',search_identity='retrieval.requests._RequestIdentity.identity',
        batch_search_action_identity='agentic.search_requests.canonical_action_identity (aggregate of canonical request identities)',
        duplicate_admission='BudgetAndGuardTracker.validate_action',requirement_support='verified_support_ids',
        evidence_local_semantic_extraction='LLMEvidenceProjector',answer_semantic_completeness='Answer LLM',
        answer_unsupported='compile_answer_requirement_coverage',answer_coverage='compile_answer_requirement_coverage',
        answer_citations='_extract_in_text_citations + selected-ID/domain validator',answer_overall='derive_overall_answer_status',
        finish_closure='Graph._assess_action; generic Guard accepts Finish without this closure check',
        requirement_semantics='ResearchPlanner',evidence_identity='EvidenceRegistry',final_evidence_selection='FinalEvidenceSelector')
    expected=dict(search_identity=('paperbase/retrieval/requests.py','identity'),
        requirement_support=('paperbase/agentic/grounding.py','verified_support_ids'),
        answer_coverage=('paperbase/generation/answer_coverage.py','compile_answer_requirement_coverage'),
        citation=('paperbase/generation/answer_generator.py','_extract_in_text_citations'),
        overall=('paperbase/generation/answer_coverage.py','derive_overall_answer_status'),
        duplicate=('paperbase/agentic/guards.py','validate_action'))
    candidates={key:[] for key in expected}
    for path in (ROOT/'paperbase').rglob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8-sig'))
        for n in ast.walk(tree):
            for key,(owner,name) in expected.items():
                if isinstance(n,ast.FunctionDef) and n.name==name:
                    candidates[key].append(dict(path=path.relative_to(ROOT).as_posix(),line=n.lineno))
    conflicts=[dict(concept=k,definitions=v) for k,v in candidates.items() if len(v)!=1 or v[0]['path']!=expected[k][0]]
    delegation=[dict(name='Selector support wrapper',passed='verified_support_ids' in inspect.getsource(selector._supported_requirement_ids)),
        dict(name='Answer bridge uses shared overall helper',passed='derive_overall_answer_status' in inspect.getsource(answering.generate_agentic_answer)),
        dict(name='preflight and formal Guard use same admission helper',passed='_assess_candidate_with_guard' in inspect.getsource(graph._decide_action) and '_assess_candidate_with_guard' in inspect.getsource(graph._validate_action)),
        dict(name='search fingerprint delegates canonical identity',passed='canonical_action_identity' in inspect.getsource(guards.action_fingerprint)),
        dict(name='typed Action adapter excludes qualifications',passed=set(schemas.AgentDecisionDraft.model_fields)=={'decision'})]
    audit=dict(checked_concepts=candidates,delegation_checks=delegation,authority_conflicts=conflicts,
        method='Scoped AST owner-definition counts and source delegation checks, then manual review of distinct responsibilities. Not a proof that arbitrary future helpers cannot conflict.',
        distinctions=['request identity vs batch action identity','Projector assertion vs verified support',
                      'Guard research-action legality vs Graph Finish closure','Classic provider contract vs Agentic semantic draft',
                      'Graph deterministic insufficient short circuit vs LLM completeness'])
    assert all(r['passed'] for r in delegation)
    from paperbase.agentic.decision_view import compute_no_progress_signal
    no_progress_keys=list(compute_no_progress_signal([]))
    return dict(prompts=prompts,schemas=schema_versions,authorities=authority,audit=audit,
        semantic_authority=dict(checkpoint_version=graph._SEMANTIC_AUTHORITY_VERSION,evaluator_version=EVALUATION_SCHEMA_VERSION,
            checkpoint_namespace=checkpoint_serde.AGENTIC_CHECKPOINT_NAMESPACE,checkpoint_serde_version='UNVERSIONED',
            projection_semantic_request_version=projection_semantics.PROJECTION_SEMANTIC_REQUEST_VERSION),
        no_progress_fields=no_progress_keys)


def observations():
    path=ROOT/'output/qwen_recommended_method002_20261006'
    manifest=json.loads((path/'run/run_manifest.json').read_text(encoding='utf-8'))
    raw=json.loads((path/'results.json').read_text(encoding='utf-8'))
    allowed=('case_id','runs','run_status','stop_reason','decision_model_calls','decision_iterations',
        'decision_prompt_tokens','decision_completion_tokens','duplicate_rejections','formal_guard_rejections',
        'invalid_iterations','tool_calls','section_context_calls','all_decision_outputs_schema_legal',
        'answer_status','answer_overall_status','answer_model_calls','sdk_create_calls')
    cases=[{k:r[k] for k in allowed if k in r} for r in raw['case_results']]
    previous=json.loads((ROOT/'output/prompt_contract_fix_live_2case_20261006/results.json').read_text(encoding='utf-8'))
    prior=[{k:r[k] for k in allowed if k in r} for r in previous['case_results']]
    return dict(status='candidate sampling baseline; not full 40-case validated',
        latest_manifest={k:manifest[k] for k in ('run_id','evaluation_schema_version','dataset_sha256','case_count','timestamp','mode','production_semantic_authority_version')},
        latest_cases=cases,sampling_profile=raw['llm_sampling_profile'],recovery_triggered=False,
        current_profile_fixed_8_case_stability_validated=False,current_profile_full_40_case_validated=False,
        independent_semantic_judge=False,prior_profile_control_observations=prior,
        evidence_sources=[dict(path=(path/'run/run_manifest.json').relative_to(ROOT).as_posix(),sha256=sha(path/'run/run_manifest.json')),
                          dict(path=(path/'results.json').relative_to(ROOT).as_posix(),sha256=sha(path/'results.json'))],
        caveat='Different runs/plans/projections/sampling; no causal parameter attribution. complete is production model status, not independent semantic truth.')


def limitations_and_drift(runtime, version_info):
    def limit(i,d,layer,m,e,s,next_step,kind='UNRESOLVED'):
        return dict(id=i,description=d,affected_layer=layer,current_mitigation=m,observed_evidence=e,severity=s,
                    planned_next_validation=next_step,kind=kind)
    limits=[
        limit('L01','Decision can repeat a visible rejected effective request; current T=1 profile recovery stability is untested.',
            'Decision / recovery','Shared Guard rejection + one bounded same-iteration replacement + cumulative invalid iteration stop.',
            'Prior T=0 strict-json-schema method_002: 5 duplicate rejections / invalid_action_limit despite full canonical feedback. Latest T=1 run has no rejection, so does not validate recovery.',
            'HIGH','Separate authorized fixed 8-case stability / saved failure-State comparison; not run by snapshot.'),
        limit('L02','Projector can return structurally valid extraction with empty proposed support; source Integrity cannot invent omitted semantic support.',
            'Projector / support establishment','Typed/domain/source validation, scoped group repair/fallback, PRE/POST diagnostics, support monotonicity.',
            'Historical structural trace confirms upstream empty FACT assertions plus a distinct domain rejection repaired for MECHANISM; no silent converter support loss found. That older T=0 run is not a frequency measurement for current profile.',
            'MEDIUM','Evaluate support establishment with current profile separately, using existing PRE/POST metadata.'),
        limit('L03','complete/partial and relevance/entailment are model semantic judgments without independent current E2 validation.',
            'Answer / Projector / evaluation','Deterministic IDs, provenance, actual citations, selected-support mapping; no program claim of semantic completeness.',
            'AnswerSemanticDraft uses complete/partial; compiler only checks allowed support and citations. SemanticAnswerJudge is Protocol/DTO only.',
            'HIGH','Calibrated independent E2/human semantic validation in a separate task.'),
        limit('L04','qwen38_recommended has only a successful single method_002 observation, not a full stability/40-case production baseline.',
            'Evaluation / model configuration','Run manifests, frozen Golden, exact request audits; distinguish candidate config from validated quality.',
            'Latest manifest case_count=1; current-profile fixed 8-case and 40-case have not been run.',
            'HIGH','Separate authorized stability suite before formal 40-case.'),
        limit('L05','Semantic near-duplicates remain legal when canonical retrieval-relevant inputs differ.',
            'Guard / Decision strategy','Exact canonical identity dedup + no-progress/targeted-support hints.',
            'Guard compares identities; regression test explicitly permits semantic-near-duplicate distinct requests. No semantic detector implemented.',
            'INFO','Measure redundant strategies separately; no forced semantic detector in current production.','DESIGN_BOUNDARY'),
        limit('L06','Sending strict=true does not prove the provider enforces every Schema keyword or prevents semantic omission.',
            'Provider structured output','Local Pydantic/domain validation, Runtime Guard, bounded repairs/fallback.',
            'Older strict requests still produced empty query/local Schema rejection; current one-case success is not enforcement proof. Native request metadata now independently verified offline.',
            'MEDIUM','Provider/schema reliability study separately, without removing local validation.'),
        limit('L07','Base OpenAI-compatible constructor ignores a true thinking_enabled flag and initializes enable_thinking=false.',
            'Base client configuration','Current .env is false; current Agentic views explicitly set thinking, so current configured calls are unaffected.',
            'load_llm_runtime_settings parses flag; OpenAICompatibleChatClient.__init__ hard-codes false.',
            'LOW','Separate base-client contract fix/regression; snapshot must not modify it.')]
    from paperbase.agentic import decision
    drift=[]
    if 'zero-temperature' in inspect.getdoc(decision._repair_client_if_needed) and runtime['node_requests']['decision']['temperature']!=0:
        drift.append(dict(id='D01',code='DOCUMENTATION_DRIFT',source=source_of(decision._repair_client_if_needed),
            description='Repair helper docstring says zero-temperature; actual helper returns the current client unchanged, and current effective T=1.',action='REPORT_ONLY'))
    doc=ROOT/'docs/architecture/agentic-research.md'
    text=doc.read_text(encoding='utf-8')
    omitted=[k for k in version_info['no_progress_fields'] if k.startswith(('consecutive_no_targeted','last_targeted')) and k not in text]
    if omitted:
        drift.append(dict(id='D02',code='DOCUMENTATION_DRIFT',source=dict(path=doc.relative_to(ROOT).as_posix(),line=next(i for i,l in enumerate(text.splitlines(),1) if 'NoProgressSignal 只统计' in l)),
            description='Current architecture enumerates only the older no-progress signals; current builder also exposes targeted-support counters.',
            missing_fields=omitted,action='REPORT_ONLY'))
    return limits,drift


def hash_inventory(model):
    categories={}
    for base,category in (('paperbase','production_python'),('app','production_python'),('eval/scripts','evaluation_tooling_python')):
        for p in (ROOT/base).rglob('*.py'):categories[p.relative_to(ROOT).as_posix()]=category
    for name in ('config.yaml','config.example.yaml','.env.example'):categories[name]='configuration_file'
    categories['eval/datasets/golden_dataset_v1_2.jsonl']='frozen_golden'
    for name in ('README.md','docs/architecture/agentic-research.md','docs/configuration.md','docs/evaluation/current-baseline.md','docs/evaluation/e1_diagnostic.md'):
        categories[name]='existing_critical_document'
    categories[Path(__file__).relative_to(ROOT).as_posix()]='snapshot_tooling'
    for name in ('snapshot_pytest_safety.py','run_snapshot_tests.py'):
        path=ROOT/'output/architecture_snapshot_20261007'/name
        if path.exists():
            categories[path.relative_to(ROOT).as_posix()]='snapshot_test_isolation_tooling'
    rows=[dict(path=p,sha256=sha(ROOT/p),category=c) for p,c in sorted(categories.items())]
    prod={r['path']:r['sha256'] for r in rows if r['category']=='production_python'}
    config={r['path']:r['sha256'] for r in rows if r['category']=='configuration_file'}
    public_base={k:v for k,v in model.items() if k in ('provider','model','endpoint_host')}
    public_base['request']=model['base_client']
    config['.env:credential-free-effective-base']=json_hash(public_base)
    return dict(hash_algorithm='SHA256',bundle_encoding='sorted relative-path TAB hex-sha256 LF; UTF-8',
        production_tree_definition='All paperbase/**/*.py and app/**/*.py, including untracked source. Tests/output excluded. eval/scripts are separately fingerprinted tooling.',
        files=rows,production_tree_sha256=bundle(prod),production_file_count=len(prod),config_bundle_sha256=bundle(config),
        credential_boundary='.env bytes/key are not included. Config bundle includes sanitized effective base metadata and template/YAML file hashes.',
        evaluation_tooling_sha256=bundle({r['path']:r['sha256'] for r in rows if r['category']=='evaluation_tooling_python'}))


def compare(old,new):
    before=json.loads(Path(old).read_text(encoding='utf-8'));after=json.loads(Path(new).read_text(encoding='utf-8'))
    keys=('source','semantic_authority','model','nodes','contracts','authorities','budgets','prompt_versions','schema_versions','known_limitations','documentation_drift','authority_conflicts')
    print(json.dumps(dict(snapshot_schema=SCHEMA,changed_sections=[k for k in keys if before.get(k)!=after.get(k)],
        production_tree_before=before.get('source',{}).get('production_tree_sha256'),production_tree_after=after.get('source',{}).get('production_tree_sha256')),
        ensure_ascii=False,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'docs/architecture/current_snapshot')
    parser.add_argument('--test-log',type=Path)
    parser.add_argument('--protected-baseline',type=Path,help='Pre-validation protected hashes for the current task')
    parser.add_argument('--compare',nargs=2,metavar=('OLD_JSON','NEW_JSON'))
    args=parser.parse_args()
    if args.compare:
        compare(*args.compare);return
    args.output=args.output.resolve()
    if args.test_log:
        args.test_log=args.test_log.resolve()
    if not args.output.is_relative_to(ROOT/'docs/architecture'):
        raise ValueError('Snapshot output must be within docs/architecture')
    before=protected_files()
    runtime=inspect_runtime();versions=versions_and_authorities();contracts=catalog()
    limits,drift=limitations_and_drift(runtime,versions);hashes=hash_inventory(runtime['model'])
    settings=runtime['settings']
    agent=settings.agentic_research
    nodes={}
    for name,req in runtime['node_requests'].items():
        ext=req['extra_body']
        nodes[name]=dict(model_call=True,temperature=req['temperature'],thinking=ext.get('enable_thinking'),
            top_p=req['top_p'],presence_penalty=req['presence_penalty'],top_k=ext.get('top_k'),min_p=ext.get('min_p'),
            repetition_penalty=ext.get('repetition_penalty'),output_mode=req['output_mode'],strict=req['strict'],
            prompt_version=versions['prompts'][name]['version'],schema_version=versions['schemas'][name]['version'],
            schema_names=versions['schemas'][name]['schema_names'],max_output_tokens_sent=req['max_tokens'],
            max_output_tokens_key_present=req['max_tokens_key_present'],timeout_seconds=req['timeout'],
            inspected_schema_sha256=req['schema_sha256'],model_call_condition='Only when node executes; cache/no-support branches can skip calls.')
    for name in ('retrieval','selector'):
        nodes[name]=dict(model_call=False,temperature=None,thinking=None,top_p=None,presence_penalty=None,top_k=None,min_p=None,
            repetition_penalty=None,output_mode=None,strict=None,prompt_version=None,schema_version=None,schema_names=None,
            max_output_tokens_sent=None,max_output_tokens_key_present=False,timeout_seconds=None,inspected_schema_sha256=None,
            model_call_condition='No generative LLM in Agentic canonical retrieval; embedding/reranker are local neural models.' if name=='retrieval' else 'Deterministic selection.')
    budgets=dict(max_agent_iterations=agent.max_agent_iterations,max_queries_per_paper=agent.max_queries_per_paper,
        node_max_attempts=agent.node_max_attempts,actual_node_attempts=runtime['actual_internal'],
        decision_recovery_replacements_per_rejected_candidate=1,invalid_counter_reset_on_valid_action=False,
        agentic_application_output_cap=None,base_client_max_tokens=runtime['model']['base_client']['max_tokens'],
        sdk_transport_default_max_retries=runtime['model']['sdk_default_max_retries'],
        answer_request_error_retries=settings.answer_generation.request_error_retries,answer_structural_citation_repair=1,
        timeouts=dict(base=runtime['model']['base_client']['timeout'],projector=agent.evidence_analysis_timeout_seconds,answer=settings.answer_generation.timeout_seconds),
        tool_top_k_per_paper=agent.top_k_per_paper,projector_batch_size=agent.evidence_projector_batch_size,
        final_evidence_max_units=agent.final_evidence_max_units,selector_cap_is_absolute=False,
        decision_view={k:v for k,v in agent.model_dump(mode='json').items() if k.startswith('decision_view')},
        compact_recent_event_window=max(agent.decision_view_recent_observations,agent.decision_view_recent_queries,agent.decision_view_recent_actions),
        classic=dict(answer_max_output_tokens=settings.answer_generation.max_output_tokens,
                     expansion_max_total_tokens=settings.context_expansion.max_total_tokens,
                     bibliography_top_k=settings.retrieval.query_rewrite.bibliography_top_k),
        retrieval=dict(dense_top_k_per_query=settings.retrieval.dense_top_k_per_query,bm25_keywords_top_k=settings.retrieval.bm25_keywords_top_k,
            fused_top_k=settings.retrieval.fused_top_k,rrf_k=settings.retrieval.rrf_k,
            reranker_candidate_top_k=settings.reranking.candidate_top_k,reranker_final_top_k=settings.reranking.final_top_k,
            retrieval_scope='Dense/FTS/RRF/reranker counts, independent of sampling top_k=20.'),
        ignored_legacy_agentic_caps=['max_output_tokens','evidence_analysis_max_output_tokens','evidence_projector_batch_base_tokens',
                                    'evidence_projector_batch_per_evidence_tokens','evidence_projector_batch_max_output_tokens'])
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip()
    dirty=bool(subprocess.run(['git','status','--porcelain'],cwd=ROOT,capture_output=True,text=True,check=True).stdout.strip())
    from paperbase.agentic import checkpoint_serde
    from paperbase.agentic.graph import AgentGraphState
    from paperbase.evaluation.contracts import SemanticAnswerJudge
    snap=dict(snapshot_schema=SCHEMA,created_at=datetime.now(timezone(timedelta(hours=8))).isoformat(),timezone='Asia/Shanghai',
        source=dict(git_commit=commit,dirty=dirty,production_tree_sha256=hashes['production_tree_sha256'],
            config_bundle_sha256=hashes['config_bundle_sha256'],curated_source_fingerprint=CURATED_TREE,
            requires_manual_architecture_review=hashes['production_tree_sha256']!=CURATED_TREE),
        semantic_authority={**versions['semantic_authority'],'golden_dataset':'eval/datasets/golden_dataset_v1_2.jsonl',
                            'golden_sha256':sha(ROOT/'eval/datasets/golden_dataset_v1_2.jsonl')},
        model=runtime['model'],nodes=nodes,contracts=contracts,authorities=versions['authorities'],budgets=budgets,
        prompt_versions=versions['prompts'],schema_versions=versions['schemas'],graph=runtime['graph'],
        no_progress_signal_fields=versions['no_progress_fields'],
        recovery=dict(shared_guard_preflight=True,formal_revalidation=True,same_iteration_bounded=True,
            ephemeral_recovery_schema_exclusion='NOT IMPLEMENTED',semantic_duplicate_detection='NOT IMPLEMENTED',
            clear_current_rejection_on=['ToolStatus.OK','ToolStatus.EMPTY','ToolStatus.PARTIAL','accepted Finish'],
            preserve_current_rejection_on=['ToolStatus.ERROR','ToolStatus.UNAVAILABLE','Guard acceptance alone'],
            rejection_history_retained=True),
        checkpoint=dict(graph_fields=list(AgentGraphState.__annotations__),
            dto_adapter_domain=[str(t).replace('paperbase.agentic.contracts.','') for t in checkpoint_serde._ADAPTERS],
            serializer='checkpoint_serde TypeAdapter JSON DTOs + SQLite SqliteSaver JsonPlusSerializer; pickle_fallback=False / allowed modules empty',
            ephemeral=['Runtime Context services/connections','DecisionState.guard_preflight callback','Compact/Factored View','CompletionView','ProjectionSemanticRequest'],
            provider_drafts_checkpointed=False,legacy_semantic_versions_parse_inspect_only=['v2','v3'],legacy_support_migration=False),
        evaluation=dict(role='E1 deterministic contracts/provenance/control diagnostics, not semantic correctness',
            golden_case_count=runtime['golden_count'],golden_taxonomy=runtime['taxonomy'],e2_status='NOT IMPLEMENTED',
            e2_protocol_source=source_of(SemanticAnswerJudge),observations=observations()),
        authority_conflicts=versions['audit']['authority_conflicts'],authority_audit=versions['audit'],
        documentation_drift=drift,known_limitations=limits,
        legacy_boundary=[dict(current=a,legacy=b) for a,b in (
            ('canonical effective search identity','raw paper/query fingerprints retained only as compatibility notes'),
            ('advisory QU hints','implicit QU retrieval input authority retired; Classic rewrite is still active'),
            ('heterogeneous scoped projections[]','singleton read accessor / historical schema, not aggregate authority'),
            ('AnswerSemanticDraft + compiler','provider coverage array archived/test-only; Classic draft active independently'),
            ('v4 execution gate','v2/v3 parse/inspect offline only'),('strict Decision json_schema','old json_object Decision retired'))])
    tests=dict(status='NOT_RUN',command='python -m pytest tests -q')
    if args.test_log:
        text=args.test_log.read_text(encoding='utf-8-sig')
        match=re.search(r'(\d+) passed, (\d+) subtests passed in ([\d.]+)s',text)
        if match and not re.search(r'\d+ failed',text):
            tests=dict(status='PASSED',passed=int(match[1]),subtests=int(match[2]),seconds=float(match[3]),
                       command='python -m pytest tests -q -p snapshot_pytest_safety',log=args.test_log.relative_to(ROOT).as_posix())
    after=protected_files()
    changed=[p for p in sorted(set(before)|set(after)) if before.get(p)!=after.get(p)]
    checks=runtime['checks']+[dict(name='no_protected_file_mutation',passed=not changed,changed_files=changed),
        dict(name='scoped_authority_audit',passed=not snap['authority_conflicts'],conflict_count=len(snap['authority_conflicts'])),
        dict(name='curated_production_fingerprint',passed=not snap['source']['requires_manual_architecture_review']),
        dict(name='full_pytest',passed=tests['status']=='PASSED',result=tests)]
    receipts=ROOT/'output/architecture_snapshot_20261007'
    baseline=args.protected_baseline.resolve() if args.protected_baseline else receipts/'protected_before.json'
    if baseline.exists():
        initial=json.loads(baseline.read_text(encoding='utf-8'))['files']
        task_changes=[p for p,expected in initial.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=expected]
        checks.append(dict(name='whole_task_original_files_unchanged',passed=not task_changes,
            original_file_count=len(initial),changed_files=task_changes,
            method='Pre-task existing source/config/tests/Golden/docs/runtime SQLite/history metadata hashes compared after inspection and full tests.'))
    if args.test_log:
        safety=json.loads((args.test_log.parent/'pytest_safety.json').read_text(encoding='utf-8'))
        test_exit=json.loads((args.test_log.parent/'pytest_exit.json').read_text(encoding='utf-8'))
        checks.append(dict(name='full_pytest_zero_network_and_storage_isolation',
            passed=test_exit['exit_code']==0 and safety['network_attempts']==0,receipt=safety,
            exit_code=test_exit['exit_code']))
    validation=dict(passed=all(c['passed'] for c in checks),provider_calls=0,network_attempts=runtime['network_attempts'],
        mock_sdk_creates=runtime['mock_sdk_creates'],checks=checks,documentation_drift_count=len(drift),
        authority_conflict_count=len(snap['authority_conflicts']),known_limitation_count=len(limits),
        inspection_boundary='No real model calls, no live eval, no Graph research execution. Local retrieval models not loaded. Schema/request metadata captured through SDK sentinel.',
        privacy=dict(credentials_stored=False,raw_prompts_stored=False,evidence_text_stored=False,golden_reference_answers_stored=False,hidden_reasoning_stored=False))
    serialized=json.dumps([snap,hashes,validation],ensure_ascii=False)
    assert runtime['runtime_secret'] not in serialized
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'architecture_snapshot.json',snap)
    write_json(args.output/'source_hashes.json',hashes)
    write_json(args.output/'snapshot_validation.json',validation)
    write_documents(args.output,snap,hashes,validation)
    bad_links=[]
    documents=list(args.output.glob('*.md'))
    for doc in documents:
        body=doc.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',body):
            path=target.split('#')[0]
            if path and not (doc.parent/path).resolve().exists():
                bad_links.append(dict(document=doc.name,target=target))
    checks.append(dict(name='snapshot_document_links',passed=not bad_links,invalid_links=bad_links))
    checks.append(dict(name='snapshot_credential_boundary',passed=bool(runtime['runtime_secret']) and
        all(runtime['runtime_secret'] not in doc.read_text(encoding='utf-8') for doc in documents),
        method='Actual configured credential compared in memory only; no secret emitted.'))
    validation['passed']=all(c['passed'] for c in checks)
    write_json(args.output/'snapshot_validation.json',validation)
    write_documents(args.output,snap,hashes,validation)
    print(json.dumps(dict(snapshot_schema=SCHEMA,tree=hashes['production_tree_sha256'],passed=validation['passed'],
        provider_calls=0,drift=len(drift),conflicts=len(snap['authority_conflicts']),limitations=len(limits)),ensure_ascii=False))


def write_documents(output,s,h,v):
    """Curated semantics from the reviewed source, with automatic factual tables."""
    write_curated_documents(output,s,h,v)


def write_curated_documents(output,s,h,v):
    def link(path,label=None):
        return f'[{label or path}](../../../{path})'
    def table(headers,rows):
        def cell(x):
            return str(x).replace('|','\\|').replace('\n','<br>')
        return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join('---' for _ in headers)+'|']+
                         ['| '+' | '.join(cell(x) for x in row)+' |' for row in rows])
    def js(value):
        return '```json\n'+json.dumps(value,ensure_ascii=False,indent=2)+'\n```'
    def save(name,text):
        (output/name).write_text(text.strip()+'\n',encoding='utf-8')
    sem=s['semantic_authority']; model=s['model']; b=s['budgets']; ev=s['evaluation']
    meta=f"Snapshot `{SCHEMA}`；生成时间 `{s['created_at']}`（Asia/Shanghai）。Commit `{s['source']['git_commit']}`；dirty=`{s['source']['dirty']}`。\n\n当前生产树 SHA256：`{h['production_tree_sha256']}`（{h['production_file_count']} 个 paperbase/app Python 文件，包含未提交文件）。配置 bundle SHA256：`{h['config_bundle_sha256']}`。历史 commit 不代表当前 dirty tree，精确路径和哈希见 [source_hashes.json](source_hashes.json)。"
    modules=[
        ('app/services/paperbase_service.py','UI Service 入口；惰性创建 Agentic/Classic 服务'),
        ('paperbase/agentic/factory.py','组装实际客户端视图、工具、分析器、Guard、Graph、checkpointer'),
        ('paperbase/agentic/scope_resolver.py','确定可信 scope / 选定论文 / KB 快照与澄清'),
        ('paperbase/agentic/query_understanding.py','意图、研究目标、advisory 初始检索提示'),
        ('paperbase/agentic/candidate_selection.py','KB metadata 候选筛选；不读取正文支持'),
        ('paperbase/agentic/planner.py','回答义务、稳定 R#、Evidence targets、投影类型'),
        ('paperbase/agentic/graph.py','Graph 节点/分支、迭代计数、Finish closure、停止与恢复生命周期'),
        ('paperbase/agentic/decision.py','Action-only structured output、本地校验、共享 preflight、有限恢复'),
        ('paperbase/agentic/decision_view.py','无 LLM 的输入去重与确定性策略信号'),
        ('paperbase/agentic/guards.py','scope、精确执行 identity 去重、预算、合法性；不生成替代 Action'),
        ('paperbase/agentic/search_requests.py','Agent Action 到低层 canonical compiler 的适配、batch 聚合 identity'),
        ('paperbase/retrieval/requests.py','实际 effective Retriever request 与逐 request identity 的编译 authority'),
        ('paperbase/retrieval','dense/FTS/加权 RRF/reranker/上下文扩展；Classic rewrite 独立'),
        ('paperbase/agentic/tools.py','接受已编译请求、执行注册工具、生成 Observation'),
        ('paperbase/agentic/evidence_admission.py','来源和 quality/reranker 特定质量门槛'),
        ('paperbase/agentic/evidence_registry.py','E#/REF# identity、provenance、bindings、assessment 合并'),
        ('paperbase/agentic/evidence_projection.py','Requirement-directed 分组投影、batch/repair/fallback、cache 校验'),
        ('paperbase/agentic/projection_semantics.py','显式 local scope 的语义请求、cache fingerprint 输入'),
        ('paperbase/agentic/grounding.py','verified_support_ids；正式局部 Requirement support authority'),
        ('paperbase/agentic/requirement_state.py','research 状态 derive / attempt tracking / finalize'),
        ('paperbase/agentic/requirement_coverage.py','critical-anchor 检索覆盖提示；不决定 Answer complete'),
        ('paperbase/agentic/selector.py','基于 required verified support 的确定性最终 Evidence 选择'),
        ('paperbase/agentic/answering.py','CompletionView 与 Agentic Answer 桥接'),
        ('paperbase/generation/answer_generator.py','语义草稿、inline citation 解析、校验/修复/降级；Classic 独立契约'),
        ('paperbase/generation/answer_coverage.py','程序编译 unsupported、实际引用支持、overall/flags'),
        ('paperbase/agentic/checkpoint_serde.py','DTO JSON adapter 与 checkpoint namespace'),
        ('paperbase/llm/client.py','provider 调用、structured format、immutable sampling views'),
        ('paperbase/evaluation','冻结 Golden 校验、E1 deterministic evaluator/audit；E2 Protocol'),
    ]
    architecture=f"""# PaperBase 当前架构快照 v1

## 一页摘要

当前模型是 `{model['model']}`，通过 `{model['endpoint_host']}` 的 ModelScope OpenAI-compatible 接口调用，Agentic 使用 `{model['sampling_profile']}`。Decision 为 temperature=1 / thinking=true，其余五个节点为 temperature=0.7 / thinking=false；六个节点都不发送应用层 max_tokens。Decision、Planner、Projector、Answer 发送 strict json_schema，QU、Candidate Selection 发送 json_object。基线客户端与 Classic 的限额和采样仍独立，详见 [配置与评测](03_config_and_evaluation.md)。

正式链为 Scope → QU →（KB 时 Candidate Selection）→ Planner → Decision/Guard/Tool 循环 → Admission → Registry → Requirement-directed 多投影 → Integrity → RequirementState → Selector → Answer semantic draft → deterministic compiler → ResearchAnswer。Guard 是 research Action runtime legality authority；Finish 研究闭合检查在 Graph。实际检索输入只由 canonical compiler 生成；QU 的英文检索词是建议。Projector 提出局部支持，`verified_support_ids` 确认正式 grounding；模型声明 complete/partial，程序编译 unsupported、实际 citations 和 overall。

checkpoint semantic authority 为 `{sem['checkpoint_version']}`；旧 v2/v3 仅允许离线 parse/inspect，不允许进入 resume 执行。E1 为 `{sem['evaluator_version']}`；Golden `{sem['golden_dataset']}` 当前 {ev['golden_case_count']} cases。E2 独立语义 Judge 尚未实现。

当前 profile 只有一次 method_002 成功 live 观察，尚无完整 8-case 稳定性或 40-case 验证。已知边界包括复杂状态下 Decision 重复选择、Projector 语义遗漏、缺乏独立 complete/partial 验证。严格结构与 grounding 不等于语义完备。此次只有只读检查和零 provider 测试，不新增 live 结果；共记录 {len(s['known_limitations'])} 项限制、{len(s['documentation_drift'])} 处文档漂移、{len(s['authority_conflicts'])} 个限定审计范围内的 authority 冲突。

## 1. Metadata 与事实来源

{meta}

事实优先级：当前 production source → 当前 config/factory 的有效组装 → DTO/schema/serde → tests → Evaluation/Golden metadata → docs。历史报告只作交叉检查。所有旧文件保持原样，漂移只记录。机器信息见 [architecture_snapshot.json](architecture_snapshot.json)，校验结果见 [snapshot_validation.json](snapshot_validation.json)。

## 2. 系统用途与入口

PaperBase 围绕本地论文知识库回答有来源的研究问题。Agentic 将问题拆成回答义务，再自主选择研究工具；Classic RAG 仍是独立检索/生成路径。UI Service 惰性创建服务，只有实际使用节点才调用模型。scope 优先可信 workspace，其次 UI paper IDs、all-KB 快照、显式论文 ID/标题/文件名；不明确或有歧义时先澄清。

当前正式 Agentic factory 面向 KnowledgeBase，未注册 workspace retrieval adapter。可信 workspace DTO 和低层适配能力存在，不代表当前 Paper Workspace UI 已接入同一 Agentic 工具链；Ask This Paper 走独立 Classic 服务。

## 3. 端到端实际调用链

```mermaid
flowchart TD
    U[User / Service] --> SC[ScopeResolver]
    SC -->|resolved| QU[Query Understanding]
    SC -->|clarification| DA[Deterministic Answer]
    QU -->|KB| CS[Candidate Selection]
    QU -->|explicit scope| P[Planner]
    CS --> P
    P --> D[Decision + local validation]
    D --> PF[Shared Guard preflight]
    PF -->|rejected / bounded replacement| D
    PF -->|candidate| AS[Graph assess_action]
    AS -->|Finish closure incomplete| D
    AS -->|research tool| C[Canonical compiler]
    C --> G[Formal Runtime Guard]
    G -->|rejection / remaining allowance| D
    G -->|legal + canonical requests| T[Tool Executor]
    T -->|search request| R[Dense / FTS / RRF / reranker]
    T -->|overview / section| NR[Non-ranked evidence]
    R --> AD[Evidence Admission]
    NR --> AD
    AD --> ER[Evidence Registry + Observation]
    ER --> EP[Scoped multi-projection + semantic cache]
    EP --> I[Integrity / verified_support_ids]
    I --> RS[RequirementState]
    RS --> D
    AS -->|accepted Finish| S[Selector]
    G -->|invalid limit| S
    D -->|iteration budget / terminal condition| S
    T -->|fatal error| S
    S --> CV[CompletionView]
    CV -->|supported selected evidence| AL[AnswerSemanticDraft LLM]
    CV -->|no usable support| DA
    AL --> AC[Deterministic Answer compiler]
    DA --> OUT[ResearchAnswer / finish_run / END]
    AC --> OUT
```

Graph 实际注册节点和全部条件边另见 JSON `graph`；此图展开了节点内部的 compiler、preflight、Integrity 与 compiler 职责。工具请求在正式执行前编译，Guard 与 Executor 使用同一请求，不是在 Retriever 内重造 query。终止通常先进入 generate_answer，再 finish_run/END，不代表每个 stop 都调用 Answer LLM。

## 4. Agent Loop 与 stop

```mermaid
flowchart TD
    START[build_plan / previous analysis] --> B[Check max iterations]
    B -->|remaining| D[Decision strict action-only JSON]
    D --> LV[Canonicalizer / local Schema + ID validation]
    LV -->|invalid / remaining candidate pool| D
    LV --> PF[Shared Guard preflight on disposable tracker]
    PF -->|reject: record current + history| RR[At most one immediate replacement]
    RR --> D
    PF -->|candidate| AS[assess_action]
    AS -->|Finish| FC[Graph candidate + required research closure]
    FC -->|not researched / invalid allowance| D
    FC -->|accepted| FINAL[Finalize RequirementState / Selector]
    AS -->|tool| FG[Formal Guard revalidation / authoritative tracker]
    FG -->|reject + allowance| D
    FG -->|accepted| EX[Execute tool]
    EX --> OBS[Admission / register_and_observe]
    OBS --> ANA[Analyze missing scoped groups / verify support]
    ANA --> STATE[Derive RequirementState]
    STATE --> B
    B -->|budget exhausted| FINAL
    FG -->|cumulative invalid limit| FINAL
    LV -->|candidate pool exhausted + invalid limit| FINAL
    EX -->|fatal error| FINAL
    FINAL --> GEN[generate_answer: selected support or deterministic no-evidence result]
    GEN --> END[finish_run / END]
```

一次成功 candidate 或耗尽 Decision candidate pool 记一个 research iteration；同 iteration 的恢复请求不另算研究轮数。cumulative invalid counter 达到 node_max_attempts 后停止，合法动作不会清零这个计数。per-paper query budget 的拒绝可重规划，并非当前代码中无条件立即停止整个 run。

`StopReason` 存在 agent_finished、budget_exhausted、per_paper_query_limit、scope_unresolved、no_available_actions、invalid_action_limit、fatal_error。不能从枚举名称推断全部可达：当前未发现 NO_AVAILABLE_ACTIONS 的产生路径；no_evidence 是状态/回答分支，不是单独 StopReason。

## 5. Evidence lifecycle

| 阶段 | Producer / output | Authority 与失败含义 |
|---|---|---|
| Retrieved | Tool/adapter → RetrievedEvidence → Observation | 原始来源与 channel/rank；空结果、unavailable、error 不等同缺乏全局答案 |
| Admitted | Evidence Admission → admitted Evidence | 按明确来源/质量过滤；仅规范化 reranker 应用对应数值门槛；拒绝并不证明语义无关 |
| Registered | EvidenceRegistry → EvidenceRecord/RegistryState | run-local identity、source provenance、requested bindings；复用合并，不丢历史请求/来源 |
| Projected | Projector → EvidenceAnalysis.projections[] | 指定局部 scope/type 下结构化抽取；有效空支持允许，遗漏不能由程序补造 |
| Verified | Integrity → verified_support_ids | 只接纳已 qualified、合法 ID、真实 source anchor、类型特定 assertion/ownership；拒绝不会自动创造支持 |
| Selected | FinalEvidenceSelector → EvidenceUnits | 依据 required verified support/新信息/质量与 rank 选择；selected 不保证实际被引用 |
| Cited | inline parser + compiler → ResearchAnswer | 只有正文真实 E#/REF#、且属于已选对应 support domain，才能进入 citations/coverage.evidence_ids |

Registry 分配正文 E# 与 bibliography REF#，R# 是 Requirement；[R1]/[REQ:R1] 不属于引用命名空间。assessment 可含多个不同 projection shape，singleton accessor 是兼容读取，不是整体 authority。新分析 scope 始终为 requested_requirement_ids ∩ current Plan IDs，显式记录 evaluated_requirement_ids；旧 scope 不猜测、不回填。

Projector 按缺失的 `(projection_type, scoped Requirement IDs)` 分组，支持 fact、mechanism、comparison、enumeration。batch 是传输聚合；semantic cache 每组独立，identity 包含 source fingerprint、问题/goal、Requirement 的 ID/description/required/targets/type、group shape、schema/prompt version。cache hit 也过当前 Integrity。只修复/降级失败同伴，保留有效同伴；成功 scope merge 保留历史有效支持，失败重分析不能抹掉先前支持。

Integrity 是局部 grounding authority，不是独立 entailment Judge。fact/mechanism/comparison 检查 proposed/supported relation 和原文 anchor；enumeration 还检查实体、角色归属与 anchor。不通过词面相似或 requirement completion 猜测提升为支持。

## 6. Requirement lifecycle

研究状态固定为 pending / in_progress / satisfied / no_support_found / uncertain。Planner 定义有序 answering obligations（最多 8 个、稳定 R1…），每项描述、required/optional、evidence_targets、projection_type 是后续语义与路由依据，不是 Retriever hard filter 或 support 判断。

未尝试为 pending；已有尝试但无支持为 in_progress；有 verified support 为 satisfied，含义仅为找到局部有效支持。正常 finish 时无支持且无不确定分析信号的已研究项进入 no_support_found；存在分析失败/不确定信号或预算结束时进入 uncertain，未触及的 pending 保持 pending。旧 terminal 状态保留（satisfied 仍按当前 verified support 重算）。只有 agent_finished/budget_exhausted 触发当前最终研究状态 finalize，invalid-limit/fatal 等停止不应被描述为正常研究闭合。候选论文 screening coverage、critical-anchor 提示、Requirement research status 和 Answer coverage 是不同轴。Decision View 用 candidate_coverage.screening_complete，不能读成回答完整。

Answer 状态 complete / partial / unsupported 是最终回答阶段。`RequirementState.satisfied != Answer complete`；Finish 要求 required 项已经研究，不要求回答 complete。Selector 不重定义 support，只接纳能支持 required Requirement 的 qualified Evidence，先做集合覆盖，再用新实体/事实、质量/provenance、section/rank/stable ID tie-break；不横向比较不同 raw score。为保存唯一 required support，phase 1 可能超出 max_units；Agentic Answer 不再次截断这些 EvidenceUnits。

## 7. Search pipeline 与 score

QU 的 semantic_query_en / lexical_keywords_en 是 advisory initial search hints，Decision 可 reuse/rewrite/translate/split/ignore。Action 中实际 query/lexical_terms 才进入 canonical compiler，QU 提示不隐式合并进 Retriever。

paper/KB 请求具有 semantic query 与显式或 fallback lexical terms；semantic whitespace 归一但保留大小写/标点。lexical 做 NFKC/casefold、去重排序、停用词过滤；非空显式 hints 替换 fallback，未显式提供时提取最多 5 词。正式 FTS 使用 quoted 单词 OR，不等同 phrase search。显式 hints 全为停用词可产生空 lexical channel，paper/KB 仍有 semantic channel；reference 此时拒绝，因为 bibliography 只有 lexical route。

逐 request identity 为 search-v1 摘要，涵盖生效 scope/backend/top-k/source/workspace 等参数；Requirement binding、evidence_targets 与无效 raw audit 字符串不参与 identity。Agentic batch action identity 是这些 canonical IDs 的排序聚合 search-action-v1，不是第二个 search request authority。Guard 区分 attempted/executed/successful/failed receipt，成功请求去重，失败/unavailable 可以合法重试。

正式检索为 Dense + FTS + weighted RRF + normalized BGE reranker 和 context expansion；Agentic canonical path 不调用生成式 query rewrite。Classic rewrite 是独立仍活跃路径。RetrievedEvidence 明确 raw_score、score_kind、score_direction、retrieval_rank：dense/reranker/RRF 各按自身方向，正式 FTS5 BM25 lower-is-better，workspace BM25 higher-is-better；overview/section 为 non-ranked、无伪造 raw score/rank。legacy score 不从正负猜语义，不存在跨 channel 校准。扩展窗口继承最佳 seed rank/channel 信息，而非比较异构原始数值。

## 8. Answer 与 recovery

Answer provider 的五个字段固定为 direct_answer、evidence_explanation、reading_interpretation、requirement_status、coverage_note。requirement_status 仅含已选 verified support 对应的 required IDs，值为 complete/partial。程序按 Plan 顺序编译全部 required coverage，对没有 selected verified support 的项给 unsupported。

正文 inline citation 是 citation source；按出现顺序去重并校验选中/对应支持 domain，编译 citations 和 coverage.evidence_ids。有支持且 complete/partial 的项至少需要一个实际引用。overall 由程序推导：没有引用/有效 coverage 或全部 unsupported 为 insufficient，全部 complete 为 complete，其余 partial，进一步推导 partial_answer/insufficient_evidence。程序不证明模型 complete/partial 的语义真值。

有一次 targeted structural/citation repair 与一次 request-error retry。可保留合法原文、必要时 complete→partial，不自动补 citations 或升级语义。JSON 无法解析且无可保留草稿时使用确定性 fallback；失败 flags 不等于问题客观不可回答。disabled/clarification/failed/no-selected-support 路径可直接生成确定性结果。case runner 保存 Answer content 调试记录，隐藏 reasoning 不导出；这些不是 checkpoint/public answer/E1 score。原有 public Evidence audit 可包含 fact/anchors，本快照不复制内容。

Decision 使用 strict Action-only root，无 reasoning/$defs/qualifications 输出字段。动态 Schema 约束可用工具/ID/domain/section pairs，但当前没有 blocked-query 的 not/const recovery exclusion。Runtime legality 由共享 Guard preflight 和 formal Guard 再验证负责；preflight 使用 disposable tracker，不另造重复算法。被拒绝后最多一次 immediate materially-different replacement，仍受现有 candidate pool/deadline 限制。

current rejection 只表示现在仍需处理的拒绝：Tool OK/EMPTY/PARTIAL 成功后或 accepted Finish 清除；Guard acceptance 本身、Tool ERROR/UNAVAILABLE 不清除。历史 rejection 保留。semantic near-duplicate detection 与 Ephemeral Recovery Schema Exclusion 均 **NOT IMPLEMENTED**。

每次 Decision 调用构建新的 state view，不携带完整聊天 transcript；Graph State 是持久 authority。soft recent events 窗口为 {b['compact_recent_event_window']}，压缩 lookup 与当前显眼 hard constraints 分开。NoProgressSignal 只作为策略提示，字段为 `{', '.join(s['no_progress_signal_fields'])}`。新增 sections/entities/reuse 不重置 support/evidence 计数；targeted support 只按新支持 IDs 与行动绑定 IDs 交集计数。无 support 不强制 finish，也没有 section 次数硬工作流。

## 9. Module map

{table(['File/module','Responsibility'],[(link(p),d) for p,d in modules])}

## 10. Evaluation 与阅读导航

E1 验证结构、source provenance、support/citation mapping、控制 trace 的确定性一致性，不判语义准确性。E2 SemanticAnswerJudge 只有 Protocol/DTO；尚无独立语义打分实现。完整 contract/authority 见 [02](02_contracts_and_authorities.md)，实际配置/版本/基线见 [03](03_config_and_evaluation.md)，真实未解决限制与漂移见 [04](04_known_limitations.md)。
"""
    save('01_current_architecture.md',architecture)
    concepts=[
        ('QueryContext','ScopeResolver + QU','可信 scope/意图与 advisory hints','Planner/Decision/Tools','canonical execution input'),
        ('ResearchPlan / ResearchRequirement','Planner','回答义务/required/投影 shape','Decision/Projector/Selector/Answer','支持成立/完整答案'),
        ('AgentAction','Decision + local validation','候选工具命令','Graph/Guard/Executor','运行合法性'),
        ('CanonicalSearchRequest','retrieval.requests compiler','生效请求与逐 request identity','Guard/Executor/Retriever','语义近重复'),
        ('RuntimeActionCapabilities','Graph State derived builder','结构可行域和 runtime hard data','Decision View/Schema','替代 formal Guard'),
        ('ActionRejectionContext','共享 preflight / formal runtime','拒绝事实和当前恢复状态','Decision/trace/checkpoint','生成新的 query'),
        ('RetrievedEvidence','Retriever/Tool','原始来源、score kind/direction/rank','Admission/Registry','跨来源统一 relevance'),
        ('EvidenceRecord','Registry','identity/provenance/binding/assessment','Integrity/State/Selector/Answer','独立语义完整性'),
        ('EvidenceProjection','Projector','指定局部 shape 的语义抽取','Integrity/Registry','自动正式 support'),
        ('verified_support_ids','grounding.py','正式局部 source-grounded support','State/Selector/CompletionView','独立 entailment/completeness'),
        ('RequirementState','derive/finalize helpers','研究尝试与局部 support 状态','Graph Finish/Decision/audit','Answer complete'),
        ('CandidateSelection','Candidate Selector','KB paper screening disposition','Guard/Coverage/Decision','Requirement answer coverage'),
        ('AnswerSemanticDraft','Answer LLM','文本和支持项 complete/partial 声明','Compiler/local validator','unsupported/API citation 数组'),
        ('AnswerRequirementCoverage','coverage compiler','required rows/actual cited mapped support/unsupported','ResearchAnswer/overall/E1','独立 complete/partial 语义裁决'),
        ('ResearchAnswer','Answer bridge + compiler / deterministic branch','最终 public 输出','Service/UI/E1','独立答案质量证明'),
    ]
    sections=[]
    for name,c in s['contracts'].items():
        rows=[]
        for variant in c['variants']:
            rows.append((variant['name'],source_link(variant['source']),'; '.join('`'+f['name']+'` ('+f['type']+')' for f in variant['core_fields'])))
        sections.append(f"### {name}\n\n用途：{c['purpose']}。Producer：{c['producer']}；Consumer：{c['consumers']}。\n\n{table(['Variant','Source','Fields'],rows)}\n\nVisibility：{c['visibility']}。checkpoint：{'是（直接或嵌套）' if c['checkpoint'] else '否（派生/临时）'}。序列化：{c['serialization']}。")
    save('02_contracts_and_authorities.md',f"""# Contracts 与 Authorities

{meta}

## Authority Matrix

{table(['Concept','Producer','Authority','Consumers','Not authority for'],concepts)}

Guard 是 research Action 的 runtime legality authority。Finish 的研究 closure 在 Graph `_assess_action`，通用 Guard 对 Finish 直接接受；两者负责不同条件，不应把 closure 的逻辑归属写错。Selector/support/Answer wrappers 复用正式 helper，不另建支持判断。低层 request identity 与 batch Action aggregate identity 也分别有明确域。

## Contract Catalog

以下字段由当前 dataclass/Pydantic/union introspection 提取；这里列 contract 字段定义，不保存实例 Evidence、真实 queries 或 provider 原文。checkpoint 标记包括嵌套于 Observation/Registry/Graph 的持久化；RuntimeActionCapabilities 是从 Graph 数据重建的 view，provider 草稿保持临时。

{chr(10).join(sections)}

## 序列化与兼容边界

Domain DTO 通过 checkpoint_serde TypeAdapter 递归 JSON 序列化，最终由 SqliteSaver 保存。不是将 provider Pydantic 草稿原样放入 checkpoint。ResearchAnswer 和 AnswerRequirementCoverage 是 public domain DTO；AnswerSemanticDraft、CompletionView、ProjectionSemanticRequest 是 internal 临时对象。

旧 Evidence.score 明确为 deprecated/legacy_unknown，不能依数值符号补猜 score kind/direction。旧 singleton projection accessor 仍可读取单投影兼容数据，但 aggregate 正式结构为 projections[]。历史 evaluated_requirement_ids 为空时保持为空；historical verified support 保留用于审计，但不能迁移到新 scope 执行。

RetrievedEvidence 的 raw_score 可为空，retrieval_rank 为同次 retrieval 的有序名次。ScoreKind 实际枚举为 dense / reranker / rrf / bm25 / bm25_fts5 / non_ranked / other / legacy_unknown；ScoreDirection 为 higher_is_better / lower_is_better / not_applicable。正式 FTS5 BM25 使用 lower_is_better，workspace BM25 使用 higher_is_better；reranker/RRF 为 higher_is_better，non_ranked 为 not_applicable。rank 只能作为既有选择信号，不是无定义的跨 channel calibration。Admission 的 min_rerank_score=0.05 只作用于明确标记 normalized 的 reranker channel。

{table(['Current','Legacy / compatibility boundary'],[(x['current'],x['legacy']) for x in s['legacy_boundary']])}

Classic 的 rewrite 和 Answer provider draft 仍在自身生产路径活跃，不能标为整个系统已删除；Agentic 已退出的仅是其旧的隐式 QU 输入/coverage 数组 authority。

## Authority conflict audit

当前 `authority_conflicts = {json.dumps(s['authority_conflicts'],ensure_ascii=False)}`。自动核对范围是 search identity、Requirement support、Answer coverage、citation、overall status、duplicate 六个指定 helper 的 AST 定义归属，加上五个 delegation/DTO 检查；人工复核各层不同责任。不存在同名竞争 owner，不等于程序已证明任意 helper 永远不会存在语义冲突。完整边界和 source locations 在 JSON `authority_audit`。

{table(['Delegation check','Result'],[(x['name'],x['passed']) for x in s['authority_audit']['delegation_checks']])}
""")
    sampling=table(['Node','T','Thinking','top_p','presence','top_k / min_p / repetition','Mode / strict','max_tokens sent','Timeout s'],[
        (name,n['temperature'],n['thinking'],n['top_p'],n['presence_penalty'],f"{n['top_k']} / {n['min_p']} / {n['repetition_penalty']}",f"{n['output_mode']} / {n['strict']}",n['max_output_tokens_sent'],n['timeout_seconds'])
        for name,n in s['nodes'].items()])
    versiontable=table(['Node','Prompt version','Schema name','Schema version'],[(name,p['version'],', '.join(s['schema_versions'][name]['schema_names']),s['schema_versions'][name]['version']) for name,p in s['prompt_versions'].items()])
    save('03_config_and_evaluation.md',f"""# Effective Config、版本、Checkpoint 与 Evaluation

{meta}

## 1. 实际客户端组装

Provider `{model['provider']}`，model `{model['model']}`，host `{model['endpoint_host']}`。只记录 endpoint host 与 credential 是否配置，不记录 API key、完整带 credential endpoint、.env 内容、Prompt 或 hidden reasoning。生产 factory 注入 SDK sentinel 后对各节点真实 client view 调用 complete_json_diagnostic，截获最终 create 参数；7 次 mock SDK create、0 次 provider。合成 schema ID 仅用于 metadata 检查，不使用用户 State，不验证 provider 真正强制 schema 的能力。

Base client（来自当前 .env 的实际 loader，隔离宿主 process override）：

{js(model['base_client'])}

Agentic `{model['sampling_profile']}` 通过 immutable view 覆盖 base 参数：

{sampling}

六个 Agentic 节点 max_tokens key 都不发送（不是发送 null），不保证 provider 没有自身输出限制。decision thinking=true 由 node view 实际设定；当前 base thinking=false。应用仍有时间、iteration、输入 view、检索/上下文限额。LLM sampling top_k=20 与 Retriever top-k 是不同参数。

Classic 未使用 Agentic profile：保持 base temperature=0.2/json_object/thinking=false（base max_tokens=1600），Classic Answer 用其自身 max_output_tokens=4000；Overview 3000、Explain 2500；query rewrite 仍按 base/client 路径。不要把 .env 温度理解为当前 Decision 温度，也不要把 Agentic 不发送 cap 理解为 Classic cap 被删除。

## 2. Budget / attempt / context

{js(b)}

实际 research max iterations=8、每 paper queries=4，node_max_attempts=2；一次拒绝最多 immediate replacement 1 次，并受剩余 candidate pool 限制。invalid iteration 累计，成功合法执行不归零。实际 Planner 内部 max_attempts=2 是 constructor default，factory 当前未显式把 node_max_attempts 传进去；两者当前相等，不意味着未来改配置自动同步。Projector group/batch repair 后可逐个 fallback，transport/node attempts/研究 iteration 不应混为同一计数。

SDK `{model['sdk_version']}` 的 transport 默认重试 {model['sdk_default_max_retries']}，没有 factory 另行覆盖；它是传输层，不等同 Answer request-error retry 或 Decision 自身候选重试。Projector/Answer timeout=60 秒，其余四个当前为 30 秒。Selector max_units=8 是选择目标，为 required-support 覆盖 phase 1 可超限；Agentic Answer 不再次截断。batch4 是 Projector transport envelope，不是一个统一语义 scope。

Decision 当前 recent_research_events 窗口=3；support scope/unresolved/section/entity 等按配置限制显示，底层 Graph State 仍保留。section metadata 与 runtime 合法 ID domain 分开。retrieval dense20/FTS20/fused50/RRF k60/reranker candidate40/final5/tool top-k5，context expansion max6000 tokens/neighbor1；此为上下文与检索边界，不是采样参数。

旧 Agentic output-cap 配置字段仍可 parse 以维持 DTO 兼容，但当前 uncapped views 不使用这些值。project.name 仍是必需 metadata，retrieval.backend 当前不驱动 route 切换，factory 固定 Hybrid；快照仅记录实际行为。

## 3. Prompt / Schema / semantic versions

{versiontable}

UNVERSIONED 表示没有显式 version constant，schema 的 name/native DTO title 不能充当版本。Prompt source SHA 和实际文本 SHA（只有哈希，没有原文）在 JSON `prompt_versions`；source_hashes 可精确区分无版本组件。Projector schema/cache version 管投影 group 结构，projection semantic request `{sem['projection_semantic_request_version']}` 管语义 fingerprint 输入，两者不是 checkpoint version。Classic Answer schema name 为 paperbase_grounded_answer。

## 4. Checkpoint acceptance 与 DTO domain

Semantic authority `{sem['checkpoint_version']}`；checkpoint namespace `{sem['checkpoint_namespace']}`；serializer 组件没有独立 version，记 `{sem['checkpoint_serde_version']}`。namespace 不等同语义版本。

| Stored semantic authority | DTO parse/inspect/roundtrip | resume / resume_stream |
|---|---|---|
| 当前 v4 | 允许 | 允许，继续正常 legality checks |
| 旧 v2 / v3 | 离线兼容允许 | 明确拒绝，在执行 graph 前 |
| 缺失 / 不匹配 | 可兼容读取的数据按 DTO 能力解析 | 明确拒绝 |

不裁剪/推断/回填旧 evaluated scope，不迁移历史 verified support。当前数据结构能 parse 不代表其旧语义可执行。

Checkpoint serializer：{s['checkpoint']['serializer']}。Graph 持久字段如下（名称来自 TypedDict）：

`{', '.join(s['checkpoint']['graph_fields'])}`

checkpoint_serde adapter domain：

{js(s['checkpoint']['dto_adapter_domain'])}

Action、canonical execution requests/receipts、QueryContext、Plan、Registry/assessment、RequirementState、Observations/current与历史rejection、Budget、Answer 等 domain 数据直接或嵌套保存。Runtime服务/连接、guard_preflight closure、Compact View、CompletionView、ProjectionSemanticRequest 不保存；provider drafts 不 checkpoint。preflight callback 每轮重新组装，不是历史可恢复 callable。

## 5. Validated deterministic baseline

Golden：{link(sem['golden_dataset'])}，SHA256 `{sem['golden_sha256']}`；冻结 {ev['golden_case_count']} cases，已从当前库以只读方式验证 metadata/ID/domain：

{table(['Primary taxonomy','Cases'],ev['golden_taxonomy'].items())}

E1 `{sem['evaluator_version']}` 负责确定性 contract、provenance、support/citation mapping、控制诊断；不是 semantic correctness 分数。E2 `SemanticAnswerJudge` 目前 **NOT IMPLEMENTED**（仅 Protocol/DTO 扩展点，未被调用）。本次生成/测试零 provider，不生成新的 live Evaluation，也不把 deterministic 全测当成 model quality baseline。

## 6. Experimental observations

当前 sampling 是 **candidate sampling baseline, not yet full 40-case validated**。最新 manifest 只跑一次 method_002；3 research iterations、3 Decision calls、2 Tool calls、0 duplicates、0 invalid、agent_finished，生产 Answer 声明 complete。这次没有触发 recovery，因此没有证明 rejection recovery 稳定。Decision prompt tokens=16311，completion=2327，total=18638。不是 8-case 或 40-case 基线，也不是 E2 confirmed complete。

下列仅保留允许的现有运行 metadata，排除了 raw trace、问题/正文/Answer 内容：

{js(ev['observations'])}

旧 T=0 strict run 的 method_002 出现重复拒绝并 invalid_action_limit；fact_001 正常。旧 Projector trace 显示上游有效空 FACT 支持与另一类可修复 MECHANISM domain error；没有证据证明 silent converter support loss。历史 Prompt/配置/Plan/检索结果不同，不能把两次差异直接归因温度或 thinking，也不能当作当前遗漏频率。

## 7. 重生成与漂移比较

先在 .venv 中进行零 provider 校验；测试使用只读生产数据与临时 SQLite 隔离（guard 脚本在 output 中，禁止 network）。

```powershell
$env:PYTHONPATH = (Get-Location).Path + ';' + (Join-Path (Get-Location) 'output/architecture_snapshot_20261007')
.venv/Scripts/python.exe -X utf8 -m pytest tests -q -p snapshot_pytest_safety
.venv/Scripts/python.exe -X utf8 scripts/generate_architecture_snapshot.py --test-log output/architecture_snapshot_20261007/pytest_full.log
.venv/Scripts/python.exe -X utf8 scripts/generate_architecture_snapshot.py --compare OLD/architecture_snapshot.json NEW/architecture_snapshot.json
```

执行时需把 pytest stdout/stderr 保存为指定 UTF-8 log，并检查 exit code（本次由 subprocess runner 完成）。不传 --test-log 时 full_pytest=NOT_RUN，validation 不会冒称通过。可用 --output 指定 docs/architecture 下另一目录作 diff。配置/版本/哈希/Schema/Graph/DTO 会重新读取；语义解释是人工复核文本，若 production tree 改变则 requires_manual_architecture_review=true、validation fail，需人工核查，不能自动批准新树。CLI __main__ wrapper 会执行入口，所以只 compile；对应 library implementation 全部 import。该界限在 validation 显式记录。
""")
    limitsections=[]
    for item in s['known_limitations']:
        limitsections.append(f"### {item['id']} — {item['description']}\n\n- Affected layer：{item['affected_layer']}\n- Severity / kind：{item['severity']} / {item['kind']}\n- Current mitigation：{item['current_mitigation']}\n- Observed evidence：{item['observed_evidence']}\n- Planned next validation：{item['planned_next_validation']}")
    driftsections=[]
    for item in s['documentation_drift']:
        driftsections.append(f"### {item['id']} — DOCUMENTATION_DRIFT\n\n{source_link(item['source'])}：{item['description']}\n\n处理：REPORT_ONLY。旧文件保持不变。")
    save('04_known_limitations.md',f"""# 当前已知限制与文档漂移

{meta}

本文件只记录当前真实边界与证据不足；没有添加生产功能/Prompt规则/采样试验。已修复的 score contract、canonical request、local-scope、多投影 cache、Answer compiler、rejection serializer 不继续列为未解决缺陷。

## 未解决限制（{len(s['known_limitations'])} 项）

{chr(10).join(limitsections)}

L01/L02 的证据来自旧采样 live 或历史保存 trace，故记录为能力风险/未验证稳定性，不能把旧失败率宣称为当前 profile 失败率。L03 的本地 anchor 校验保证原文来源与 domain，不证明语义蕴含。L05 是精确去重的设计边界，不是已承诺而缺失的 semantic Judge。L07 是 base constructor 的潜在配置问题，当前 false base 与显式 Agentic views 不受影响。

## DOCUMENTATION_DRIFT（{len(s['documentation_drift'])} 处）

{chr(10).join(driftsections)}

## Authority audit 与检查范围

本次六项核心 owner/helper + delegation 审计报告 {len(s['authority_conflicts'])} 个冲突，JSON `authority_conflicts` 为 `{json.dumps(s['authority_conflicts'],ensure_ascii=False)}`。这是限定检查结果，不是全系统语义正确性的证明。Finish closure 在 Graph、candidate screening 与 Answer completeness、request/batch identity 和 Classic/Agentic 合约均明确区分。

观察来源只读取 {link('output/qwen_recommended_method002_20261006/run/run_manifest.json')}、{link('output/qwen_recommended_method002_20261006/results.json')}、{link('output/prompt_contract_fix_live_2case_20261006/results.json')}、{link('output/projector_structural_trace_20261006/projector_structural_trace_report.md')} 的既有记录与当前源码，不导出其中 Evidence/Answer 内容。未来 next validation 只是建议，未被本轮自动授权或执行。

## 本轮校验

Validation passed=`{v['passed']}`，provider calls=0；{len(v['checks'])} 项检查详情见 [snapshot_validation.json](snapshot_validation.json)。本轮原始 production/config/tests/Golden/docs/checkpoint 的哈希前后核对，测试对生产 SQLite 的写入隔离到临时目录。安全隔离只属于本次 tooling，未改变 Graph/checkpoint 生产实现。
""")


if __name__=='__main__':
    main()
