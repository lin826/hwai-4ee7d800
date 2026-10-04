from pathlib import Path

import pytest

from health_context.agent import answer
from health_context.index import PatientIndex
from health_context.llm import Reply, ToolCall, parse_pythonic_calls
from health_context.pipeline import load_bundle

DATA = Path(__file__).resolve().parents[1] / "data"
SAMPLE = next(DATA.glob("Merlene950_Marlin805_Thompson596_*.json"), None)
LATEST_HBA1C = "f5749532-3295-ad01-c588-daa4b0203574"


def test_parse_lfm_tool_call_block():
    text = '<|tool_call_start|>[get_timeline(concept_id="lab:4548-4", since="2020-01-01")]<|tool_call_end|>Checking.'
    calls, rest = parse_pythonic_calls(text)
    assert [(c.name, c.arguments) for c in calls] == [
        ("get_timeline", {"concept_id": "lab:4548-4", "since": "2020-01-01"})
    ]
    assert rest == "Checking."


def test_parse_bare_call_list_and_plain_text():
    calls, _ = parse_pythonic_calls('[search_records(query="statin", k=3)]')
    assert calls[0].arguments == {"query": "statin", "k": 3}
    assert parse_pythonic_calls("Her latest HbA1c was 6.31 %.") == (
        [],
        "Her latest HbA1c was 6.31 %.",
    )


class ScriptedClient:
    """Stands in for the LLM server: one tool call, then a final answer."""

    model = "scripted"

    def __init__(self):
        self.requests = []

    def chat(self, messages, tools=None):
        self.requests.append(messages)
        if len(self.requests) == 1:
            return Reply(
                "", [ToolCall("c1", "get_timeline", {"concept_id": "lab:4548-4"})]
            )
        tool_output = messages[-1]["content"]
        assert f"Observation/{LATEST_HBA1C}" in tool_output
        return Reply(f"6.31 % on 2025-09-29 (Observation/{LATEST_HBA1C})")


@pytest.mark.skipif(SAMPLE is None, reason="extract data/ first (see DATA.md)")
def test_agent_runs_tool_then_answers():
    client = ScriptedClient()
    result = answer(PatientIndex(load_bundle(SAMPLE)), "Latest HbA1c?", client)
    assert result.tool_calls == 1
    assert LATEST_HBA1C in result.text
    first_user = client.requests[0][1]["content"]
    assert "[lab:4548-4]" in first_user and "Question: Latest HbA1c?" in first_user
