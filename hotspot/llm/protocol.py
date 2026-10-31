"""Shared types for model clients."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class ModelReply:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    raw: dict = field(default_factory=dict)


class ModelClient(Protocol):
    name: str

    def complete(
        self,
        system: str,
        messages: list[dict],
        images: list[bytes] | None,
        max_tokens: int,
        temperature: float,
    ) -> ModelReply: ...
