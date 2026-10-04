"""Minimal client for any OpenAI-compatible chat server (Ollama, llama.cpp, vLLM).

The default target is a locally hosted LiquidAI LFM2.5-8B-A1B served by
Ollama. Only the standard library is used, so switching servers is a matter
of ``LLM_BASE_URL`` and ``LLM_MODEL``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "hf.co/LiquidAI/LFM2.5-8B-A1B-GGUF:Q8_0"


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Reply:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)


class ChatClient:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        timeout: float = 300,
    ):
        self.base_url = (
            base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or DEFAULT_MODEL
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or "local"
        self.timeout = timeout

    def chat(
        self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None
    ) -> Reply:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if tools:
            body["tools"] = tools
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = json.load(response)
        message = raw["choices"][0]["message"]
        content = message.get("content") or ""
        calls = [
            ToolCall(
                c.get("id") or f"call_{i}",
                c["function"]["name"],
                _arguments(c["function"].get("arguments")),
            )
            for i, c in enumerate(message.get("tool_calls") or [])
        ]
        if not calls:
            # LFM2.5 writes Pythonic calls in the text when the server's
            # template does not translate them into structured tool_calls.
            calls, content = parse_pythonic_calls(content)
        return Reply(content.strip(), calls, raw)


def _arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    return json.loads(value)


_CALL_BLOCK = re.compile(r"<\|tool_call_start\|>(.*?)<\|tool_call_end\|>", re.DOTALL)
_BARE_LIST = re.compile(r"^\s*(\[\s*[A-Za-z_]\w*\(.*\)\s*\])\s*$", re.DOTALL)


def parse_pythonic_calls(text: str) -> tuple[list[ToolCall], str]:
    """Extract ``[name(arg="v"), ...]`` calls; return them and the remaining text."""
    blocks = _CALL_BLOCK.findall(text)
    remaining = _CALL_BLOCK.sub("", text)
    if not blocks:
        bare = _BARE_LIST.match(text)
        if not bare:
            return [], text
        blocks, remaining = [bare.group(1)], ""
    calls: list[ToolCall] = []
    for block in blocks:
        try:
            tree = ast.parse(block.strip(), mode="eval").body
        except SyntaxError:
            continue
        nodes = tree.elts if isinstance(tree, ast.List) else [tree]
        for node in nodes:
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                try:
                    args = {
                        kw.arg: ast.literal_eval(kw.value)
                        for kw in node.keywords
                        if kw.arg
                    }
                except ValueError:
                    continue
                calls.append(ToolCall(f"call_{len(calls)}", node.func.id, args))
    return calls, remaining


def openai_tools(schemas: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert ``tools.TOOL_SCHEMAS`` into the OpenAI ``tools`` request format."""
    return [
        {
            "type": "function",
            "function": {
                "name": s["name"],
                "description": s["description"],
                "parameters": s["input_schema"],
            },
        }
        for s in schemas
    ]
