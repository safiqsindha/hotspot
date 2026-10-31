"""Canned replies so agent tests need no network."""

from __future__ import annotations

from hotspot.llm.protocol import ModelReply


class ScriptedClient:
    name = "scripted"
    model = "scripted"

    def __init__(self, replies: list[str], input_tokens: int = 11, output_tokens: int = 7):
        self._replies = list(replies)
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[dict] = []

    def complete(self, system, messages, images=None, max_tokens=300, temperature=0.0) -> ModelReply:
        self.calls.append(
            {
                "system": system,
                "messages": messages,
                "n_images": len(images or []),
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        text = self._replies.pop(0) if self._replies else ""
        return ModelReply(
            text=text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            latency_s=0.0,
            raw={"text": text},
        )
