"""Answer one question about one patient: catalog in the prompt, tools for detail."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .index import PatientIndex
from .llm import ChatClient, openai_tools
from .tools import TOOL_SCHEMAS, catalog, run_tool

SYSTEM_PROMPT = """You answer questions about one patient's synthetic medical record.

The user message starts with the patient's catalog: every recorded concept with its count, date span and latest value. Concept ids are in [brackets].
- For exact values, dates, trends or anything not shown in the catalog, call get_timeline with a concept id, or search_records first if you need to find one.
- Answer only from the catalog and tool results. Give values with units and dates as YYYY-MM-DD.
- Cite the FHIR resource behind each fact as ResourceType/id, copied from a tool result.
- If something is not in the record, say it is "not recorded". Never infer that it is absent or invent it.
Keep the final answer short."""


@dataclass
class Answer:
    text: str
    steps: list[dict] = field(default_factory=list)

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s["type"] == "tool")


def answer(
    index: PatientIndex, question: str, client: ChatClient, max_steps: int = 6
) -> Answer:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{catalog(index)}\n\nQuestion: {question}"},
    ]
    tools = openai_tools(TOOL_SCHEMAS)
    steps: list[dict] = []
    for _ in range(max_steps):
        reply = client.chat(messages, tools)
        if not reply.tool_calls:
            return Answer(reply.content, steps)
        messages.append(
            {
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.name,
                            "arguments": json.dumps(c.arguments),
                        },
                    }
                    for c in reply.tool_calls
                ],
            }
        )
        for call in reply.tool_calls:
            try:
                result = run_tool(index, call.name, call.arguments)
            except (KeyError, TypeError, ValueError) as error:
                result = f"Tool error: {error!r}. Check the arguments."
            steps.append(
                {
                    "type": "tool",
                    "name": call.name,
                    "arguments": call.arguments,
                    "result_chars": len(result),
                }
            )
            messages.append(
                {"role": "tool", "tool_call_id": call.id, "content": result}
            )
    # Out of steps: ask for an answer from what was gathered, without tools.
    messages.append(
        {"role": "user", "content": "Answer now from the information above."}
    )
    return Answer(client.chat(messages).content, steps)
