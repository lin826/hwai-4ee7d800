"""Authoritative context size: Anthropic's token counting endpoint, as the README specifies.

A context is counted as one user message on ``claude-opus-5``. The key is used
only for ``messages.count_tokens``; this module never requests a completion.
"""

from __future__ import annotations

import os
from pathlib import Path

COUNT_MODEL = "claude-opus-5"
MAX_TOKENS = 900_000


def load_dotenv(path: Path | str = ".env") -> None:
    """Read KEY=VALUE lines into the environment without overriding set variables."""
    env = Path(path)
    if not env.is_file():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(
            key.strip().removeprefix("export "), value.strip().strip("\"'")
        )


class AnthropicCounter:
    """Callable ``text -> input tokens`` for one user message on ``claude-opus-5``."""

    method = f"anthropic_count_tokens:{COUNT_MODEL}"

    def __init__(self, model: str = COUNT_MODEL):
        load_dotenv()
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is not set (environment or .env)")
        from anthropic import Anthropic

        self.client = Anthropic()
        self.model = model

    def __call__(self, text: str) -> int:
        return self.client.messages.count_tokens(
            model=self.model,
            messages=[{"role": "user", "content": text}],
        ).input_tokens
