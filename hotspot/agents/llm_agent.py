"""One model call per environment step. Replies are a single JSON action."""

from __future__ import annotations

import io
import json
import re

from hotspot.schema import VERBS, World

SYSTEM = (
    "You play a point-and-click adventure. Reply with one JSON object and nothing else. "
    "Do not explain. Verbs: " + ", ".join(VERBS) + "."
)


def goal_text(world: World) -> str:
    if world.goal.description:
        return world.goal.description
    return world.title


def parse_action(text: str) -> dict | None:
    if not text:
        return None
    stripped = text.strip()
    stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
    stripped = re.sub(r"\s*```$", "", stripped)
    for candidate in (stripped, text):
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            value = _first_object(candidate)
        if isinstance(value, dict) and isinstance(value.get("verb"), str):
            return value
    return None


def _first_object(text: str):
    start = text.find("{")
    if start < 0:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value


def build_user_prompt(world: World, observation, history: list[dict]) -> str:
    """Text the model sees. No puzzle graph, polygons, optimal path, or hidden items."""
    lines = [
        f"Goal: {goal_text(world)}",
        f"Scene: {observation.scene}",
        "Inventory: " + (", ".join(observation.inventory) if observation.inventory else "(empty)"),
        f"Last result: {observation.text or '(none)'}",
    ]
    if history:
        lines.append("Recent actions:")
        for entry in history:
            lines.append(f"- {json.dumps(entry['action'], sort_keys=True)} -> {entry['result']}")
    if getattr(observation, "mode", "symbolic") == "pixel":
        lines.append(
            "You see only the image. Hotspot names are not listed. "
            'Click with {"verb":"take","x":0.5,"y":0.5} or {"verb":"use","x":0.5,"y":0.5,"item":"ITEM"}. '
            'Zoom with {"verb":"zoom","box":[0.2,0.2,0.5,0.5]}. Coordinates are 0 to 1.'
        )
    else:
        lines.append("Visible hotspots:")
        for hotspot in observation.hotspots:
            verbs = ", ".join(hotspot["verbs"])
            lines.append(f"- {hotspot['name']} [{hotspot['id']}]: {verbs}")
        lines.append(
            'Reply with {"verb":"take","target":"HOTSPOT_ID"} '
            'or {"verb":"use","target":"HOTSPOT_ID","item":"ITEM"}.'
        )
    return "\n".join(lines)


class LLMAgent:
    def __init__(self, client, mode: str = "symbolic", world: World | None = None, *, memory: int = 8, temperature: float = 0.0, seed: int | None = None, max_tokens: int = 400):
        if mode not in ("symbolic", "pixel"):
            raise ValueError("mode must be 'symbolic' or 'pixel'")
        self.client = client
        self.mode = mode
        self.world = world
        self.memory = memory
        self.temperature = temperature
        self.seed = seed
        self.max_tokens = max_tokens
        self.model_name = getattr(client, "model", None) or getattr(client, "name", "llm")
        self._history: list[dict] = []
        self._pending: dict | None = None
        self.last_step_meta: dict = {}

    def act(self, observation) -> dict:
        if self._pending is not None:
            self._history.append({"action": self._pending, "result": observation.text or ""})
            self._history = self._history[-self.memory :]
        world = self.world
        if world is None:
            prompt = f"Goal: Solve the scene.\nScene: {observation.scene}\nInventory: (empty)"
        else:
            prompt = build_user_prompt(world, observation, self._history[-self.memory :])
        images = _image_bytes(observation) if self.mode == "pixel" else None
        messages = [{"role": "user", "content": prompt}]
        retries = 0
        input_tokens = 0
        output_tokens = 0
        latency = 0.0
        last_text = ""
        action = None
        while True:
            reply = self.client.complete(
                SYSTEM,
                messages,
                images if retries == 0 else None,
                self.max_tokens,
                self.temperature,
            )
            input_tokens += reply.input_tokens
            output_tokens += reply.output_tokens
            latency += reply.latency_s
            last_text = reply.text
            action = parse_action(reply.text)
            if action is not None or retries >= 2:
                break
            retries += 1
            messages.append({"role": "assistant", "content": reply.text})
            messages.append({"role": "user", "content": "Reply with one JSON object only."})
        if action is None:
            action = {"verb": "look"}
        self.last_step_meta = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency_s": latency,
            "raw_reply": last_text,
            "parse_retries": retries,
            "model": self.model_name,
            "temperature": self.temperature,
            "seed": self.seed,
        }
        self._pending = action
        return action


def _image_bytes(observation) -> list[bytes] | None:
    image = getattr(observation, "image", None)
    if image is None:
        return None
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return [buffer.getvalue()]
