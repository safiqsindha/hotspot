"""Anthropic Messages API. The key is read from the environment and never logged."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from base64 import b64encode

from hotspot.llm.protocol import ModelReply


class AnthropicClient:
    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self.name = model
        self._api_key = api_key or os.environ["ANTHROPIC_API_KEY"]

    def __repr__(self) -> str:
        return f"AnthropicClient(model={self.model!r})"

    def complete(self, system, messages, images=None, max_tokens=300, temperature=0.0) -> ModelReply:
        content_messages = []
        for index, message in enumerate(messages):
            content = message["content"]
            if images and index == 0 and message.get("role") == "user":
                blocks = []
                for blob in images:
                    blocks.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64encode(blob).decode("ascii"),
                            },
                        }
                    )
                blocks.append({"type": "text", "text": content})
                content_messages.append({"role": "user", "content": blocks})
            else:
                content_messages.append({"role": message["role"], "content": content})
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "system": system,
            "messages": content_messages,
        }
        request = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": self._api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"anthropic HTTP {exc.code}: {detail}") from None
        text = "".join(
            block.get("text", "")
            for block in payload.get("content", [])
            if isinstance(block, dict) and block.get("type") == "text"
        )
        usage = payload.get("usage") or {}
        return ModelReply(
            text=text,
            input_tokens=int(usage.get("input_tokens") or 0),
            output_tokens=int(usage.get("output_tokens") or 0),
            latency_s=time.perf_counter() - started,
            raw={"id": payload.get("id"), "model": payload.get("model"), "stop_reason": payload.get("stop_reason")},
        )
