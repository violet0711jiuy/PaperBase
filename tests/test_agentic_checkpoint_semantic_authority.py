"""PR-4B compatibility-only patch: versioned execution, lossless offline inspection."""
from copy import deepcopy
import json

from langgraph.checkpoint.memory import InMemorySaver
from pydantic import TypeAdapter
import pytest

from paperbase.agentic.checkpoint_serde import checkpoint_thread_id, dump_contract, load_contract
from paperbase.agentic.contracts import GuardState, QueryContext, RegistryState, ResearchObservation, ResearchPlan
from paperbase.agentic.decision import DecisionState
from paperbase.agentic.graph import _SEMANTIC_AUTHORITY_VERSION
from paperbase.agentic.grounding import verified_support_ids
from test_agentic_langgraph import _EvidenceAwareDecisionAgent, _Executor, _workflow
from test_evidence_target_contract import HISTORY


OLD_VERSION = "projector-integrity-answer-support-boundary-v2"
NEW_VERSION = "projector-requirement-directed-multi-projection-v4"


@pytest.fixture(autouse=True)
def forbid_llm(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Checkpoint semantic authority tests must not call an LLM/provider")
    from paperbase.agentic import query_understanding, planner, decision, evidence_projection
    for module in (query_understanding, planner, decision, evidence_projection):
        monkeypatch.setattr(module, "complete_json_with_diagnostics", forbidden)


def config(thread_id):
    return {"configurable": {"thread_id": checkpoint_thread_id(thread_id)}}


def interrupted_workflow(thread_id="authority", *, completed=False):
    executor = _Executor()
    saver = InMemorySaver()
    workflow = _workflow(_EvidenceAwareDecisionAgent(), executor, checkpointer=saver)
    execution = workflow.start("A 和 B 的方法有什么区别？", selected_paper_ids=("A", "B"),
        thread_id=thread_id, interrupt_after=() if completed else ("execute_tool",))
    return workflow, executor, saver, execution


def set_v2(workflow, thread_id, *, completed=False):
    workflow._graph.update_state(config(thread_id), {"semantic_authority_version": OLD_VERSION},
        as_node="finish_run" if completed else "execute_tool")


def assert_refused(workflow, thread_id, mode):
    with pytest.raises(ValueError, match="Incompatible checkpoint semantic authority") as failure:
        if mode == "resume":
            workflow.resume(thread_id)
        else:
            list(workflow.resume_stream(thread_id))  # generators validate when consumed
    assert OLD_VERSION in str(failure.value)
    assert NEW_VERSION in str(failure.value)
    assert "start a new research thread" in str(failure.value)


def historical_states():
    adapter = TypeAdapter(DecisionState)
    for path in HISTORY:
        for line in path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            yield path, row, adapter.validate_python(row["full_decision_state"])


def install_historical_v2(workflow, thread_id, state):
    workflow._graph.update_state(config(thread_id), {
        "semantic_authority_version": OLD_VERSION, "thread_id": thread_id,
        "original_query": state.context.original_query,
        "query_context": dump_contract(state.context, QueryContext),
        "plan": dump_contract(state.plan, ResearchPlan),
        "registry_state": dump_contract(RegistryState(records=state.evidence), RegistryState),
        "observations": dump_contract(state.observations, tuple[ResearchObservation, ...]),
        "guard_state": dump_contract(GuardState(), GuardState),
        "next_node": "analyze_evidence",
    }, as_node="register_and_observe")


def test_fresh_thread_persists_v4_authority():
    workflow, _, _, execution = interrupted_workflow()
    assert _SEMANTIC_AUTHORITY_VERSION == NEW_VERSION != OLD_VERSION
    assert execution.interrupted
    assert execution.state["semantic_authority_version"] == NEW_VERSION
    assert workflow.inspect("authority").state["semantic_authority_version"] == NEW_VERSION


@pytest.mark.parametrize("mode", ["resume", "resume_stream"])
def test_v4_interrupt_resume_works_across_workflow_reconstruction(mode):
    _, first_executor, saver, execution = interrupted_workflow()
    assert execution.next_nodes == ("filter_evidence",)
    assert len(first_executor.calls) == 1
    next_executor = _Executor()
    workflow = _workflow(_EvidenceAwareDecisionAgent(), next_executor, checkpointer=saver)
    if mode == "resume":
        completed = workflow.resume("authority")
        assert not completed.interrupted and completed.result is not None
    else:
        events = list(workflow.resume_stream("authority"))
        assert events[0].stage == "resume"
        assert events[-1].kind == "completed" and events[-1].result is not None
    assert [action.searches[0].paper_id for action in next_executor.calls] == ["B"]
    inspected = workflow.inspect("authority")
    assert inspected.state["semantic_authority_version"] == NEW_VERSION
    assert not inspected.interrupted


@pytest.mark.parametrize("mode", ["resume", "resume_stream"])
@pytest.mark.parametrize("completed", [False, True])
def test_v2_checkpoint_is_inspectable_but_refused_before_any_graph_execution(mode, completed, monkeypatch):
    workflow, executor, _, _ = interrupted_workflow(completed=completed)
    set_v2(workflow, "authority", completed=completed)
    before = deepcopy(workflow.inspect("authority"))
    calls_before = len(executor.calls)
    def forbidden(*args, **kwargs):
        raise AssertionError("Legacy v2 must not reach invoke/stream")
    monkeypatch.setattr(workflow._graph, "invoke", forbidden)
    monkeypatch.setattr(workflow._graph, "stream", forbidden)
    assert_refused(workflow, "authority", mode)
    assert workflow.inspect("authority") == before
    assert len(executor.calls) == calls_before
    assert before.state["semantic_authority_version"] == OLD_VERSION


def test_thirteen_historical_states_remain_lossless_offline_but_cannot_execute(monkeypatch):
    adapter = TypeAdapter(DecisionState)
    executor = _Executor()
    workflow = _workflow(_EvidenceAwareDecisionAgent(), executor)
    def forbidden(*args, **kwargs):
        raise AssertionError("Historical v2 cannot enter current execution")
    monkeypatch.setattr(workflow._graph, "invoke", forbidden)
    monkeypatch.setattr(workflow._graph, "stream", forbidden)
    count = 0
    outside_local_scope = []
    for path, row, state in historical_states():
        original = deepcopy(state)
        assert adapter.validate_json(adapter.dump_json(state)) == state
        registry = RegistryState(records=state.evidence)
        assert load_contract(dump_contract(registry, RegistryState), RegistryState) == registry
        before_support = {r.evidence_id: verified_support_ids(r, state.plan.requirements) for r in state.evidence}
        assert all(r.assessment.evaluated_requirement_ids == () for r in state.evidence)
        thread_id = f"historical-v2-{count}"
        install_historical_v2(workflow, thread_id, state)
        snapshot = deepcopy(workflow.inspect(thread_id))
        assert snapshot.interrupted  # would route to current analysis without the version gate
        restored = load_contract(snapshot.state["registry_state"], RegistryState)
        assert restored == registry
        assert before_support == {r.evidence_id: verified_support_ids(r, state.plan.requirements) for r in restored.records}
        assert all(r.assessment.evaluated_requirement_ids == () for r in restored.records)
        for record in restored.records:
            desired = set(record.requested_requirement_ids) & {r.requirement_id for r in state.plan.requirements}
            if set(before_support[record.evidence_id]) - desired:
                outside_local_scope.append((path.parent.name, row["iteration"], record.evidence_id))
        for mode in ("resume", "resume_stream"):
            assert_refused(workflow, thread_id, mode)
        assert workflow.inspect(thread_id) == snapshot
        assert state == original
        count += 1
    assert count == 13
    assert outside_local_scope  # keep the real out-of-scope legacy support for offline audit
    assert executor.calls == []
