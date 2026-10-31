"""OpenAI-compatible chat completions. Covers OpenAI and xAI via base_url."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from base64 import b64encode

from hotspot.llm.protocol import ModelReply


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        api_key_env: str = "OPENAI_API_KEY",
    ):
        self.model = model
        self.name = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key or os.environ[api_key_env]

    def __repr__(self) -> str:
        return f"OpenAICompatibleClient(model={self.model!r}, base_url={self.base_url!r})"

    def complete(self, system, messages, images=None, max_tokens=300, temperature=0.0) -> ModelReply:
        payload_messages = [{"role": "system", "content": system}]
        for index, message in enumerate(messages):
            content = message["content"]
            if images and index == 0 and message.get("role") == "user":
                blocks = [{"type": "text", "text": content}]
                for blob in images:
                    encoded = b64encode(blob).decode("ascii")
                    blocks.append(
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded}"}}
                    )
                payload_messages.append({"role": "user", "content": blocks})
            else:
                payload_messages.append({"role": message["role"], "content": content})
        body = {
            "model": self.model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "messages": payload_messages,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(f"openai-compatible HTTP {exc.code}: {detail}") from None
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content") or ""
        usage = payload.get("usage") or {}
        return ModelReply(
            text=text,
            input_tokens=int(usage.get("prompt_tokens") or 0),
            output_tokens=int(usage.get("completion_tokens") or 0),
            latency_s=time.perf_counter() - started,
            raw={"id": payload.get("id"), "model": payload.get("model")},
        )


def xai_client(model: str, api_key: str | None = None) -> OpenAICompatibleClient:
    return OpenAICompatibleClient(
        model,
        api_key=api_key,
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
    )
