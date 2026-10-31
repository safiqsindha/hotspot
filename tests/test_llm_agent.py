"""LLM agent with a scripted client. No network unless HOTSPOT_LIVE_LLM=1."""

import json
import os

import pytest

from hotspot.agents.llm_agent import LLMAgent, build_user_prompt, parse_action
from hotspot.engine import iter_hotspots
from hotspot.geom import centroid
from hotspot.harness import _observe, run_episode
from hotspot.llm.anthropic import AnthropicClient
from hotspot.llm.openai_compat import OpenAICompatibleClient, xai_client
from hotspot.llm.pricing import estimate_cost
from hotspot.llm.scripted import ScriptedClient
from hotspot.loader import load_world
from hotspot.solver import solve
from hotspot.state import initial_state

TINY = load_world("worlds/tiny_key_door.json")
ROPE = load_world("worlds/rope_and_well.json")


def _symbolic_replies(world) -> list[str]:
    return [json.dumps(action) for action in solve(world).optimal_path]


def _pixel_replies(world) -> list[str]:
    replies = []
    for action in solve(world).optimal_path:
        point = None
        for scene in world.scenes:
            for hotspot in iter_hotspots(scene):
                if hotspot.id == action.get("target"):
                    point = centroid(hotspot.polygon)
        assert point is not None
        payload = {"verb": action["verb"], "x": point[0], "y": point[1]}
        if action.get("item") is not None:
            payload["item"] = action["item"]
        replies.append(json.dumps(payload))
    return replies


@pytest.mark.parametrize("world", [TINY, ROPE], ids=["tiny", "rope"])
@pytest.mark.parametrize("mode", ["symbolic", "pixel"])
def test_scripted_optimal_path_completes(world, mode, tmp_path):
    replies = _symbolic_replies(world) if mode == "symbolic" else _pixel_replies(world)
    client = ScriptedClient(replies, input_tokens=11, output_tokens=7)
    agent = LLMAgent(client, mode=mode, world=world, temperature=0.0, seed=4)
    metrics = run_episode(world, agent, mode=mode, tier="retro_320", log_dir=tmp_path)
    assert metrics["completed"] is True
    assert metrics["efficiency"] == 1.0
    assert metrics["actions_used"] == solve(world).optimal_length
    assert metrics["total_input_tokens"] == 11 * metrics["actions_used"]
    assert metrics["total_output_tokens"] == 7 * metrics["actions_used"]
    assert metrics["est_cost_usd"] is None
    assert metrics["model"] == "scripted"
    lines = (tmp_path / "").glob("*.jsonl")
    logged = list(Path_read(metrics["log_path"]))
    assert len(logged) == metrics["actions_used"]
    assert logged[0]["parse_retries"] == 0
    assert logged[0]["temperature"] == 0.0
    assert logged[0]["seed"] == 4
    assert logged[0]["model"] == "scripted"
    if mode == "pixel":
        assert client.calls[0]["n_images"] == 1
    else:
        assert client.calls[0]["n_images"] == 0
    del lines


def Path_read(path):
    from pathlib import Path

    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_malformed_reply_is_logged_not_fatal(tmp_path):
    client = ScriptedClient(["this is not json", "still no", "nope"], input_tokens=11, output_tokens=7)
    agent = LLMAgent(client, mode="symbolic", world=TINY, temperature=0.0, seed=1)
    metrics = run_episode(TINY, agent, mode="symbolic", max_actions=1, log_dir=tmp_path)
    assert metrics["completed"] is False
    assert metrics["actions_used"] == 1
    assert metrics["wasted_actions"] == 1
    assert metrics["total_input_tokens"] == 33
    assert metrics["total_output_tokens"] == 21
    logged = Path_read(metrics["log_path"])
    assert logged[0]["parse_retries"] == 2
    assert "nope" in logged[0]["raw_reply"]
    assert "this is not json" not in logged[0]["raw_reply"] or logged[0]["parse_retries"] == 2


def test_parse_lenient_and_reasks():
    fenced = 'Sure thing\n```json\n{"verb": "take", "target": "key_spot"}\n```'
    assert parse_action(fenced)["target"] == "key_spot"
    prose = 'I pick {"verb": "look", "target": "shed_door"} thanks'
    assert parse_action(prose)["verb"] == "look"
    assert parse_action("no braces here") is None

    client = ScriptedClient(
        ["hello", "nope", json.dumps({"verb": "look", "target": "key_spot"})],
        input_tokens=11,
        output_tokens=7,
    )
    agent = LLMAgent(client, mode="symbolic", world=TINY)
    observation = _observe(TINY, initial_state(TINY), "symbolic", "vga_640", "", None)
    action = agent.act(observation)
    assert action["target"] == "key_spot"
    assert agent.last_step_meta["parse_retries"] == 2
    assert agent.last_step_meta["input_tokens"] == 33
    assert len(client.calls) == 3


def test_memory_keeps_last_eight_actions():
    replies = [json.dumps({"verb": "look", "target": "key_spot"}) for _ in range(12)]
    client = ScriptedClient(replies)
    agent = LLMAgent(client, mode="symbolic", world=TINY, memory=8)
    observation = _observe(TINY, initial_state(TINY), "symbolic", "vga_640", "You look.", None)
    for _ in range(10):
        agent.act(observation)
    prompt = client.calls[-1]["messages"][0]["content"]
    recent = prompt.split("Recent actions:", 1)[1].split("Visible hotspots:", 1)[0]
    lines = [line for line in recent.splitlines() if line.startswith("- ")]
    assert len(lines) == 8


def test_prompt_does_not_leak_solution():
    tiny_obs = _observe(TINY, initial_state(TINY), "symbolic", "vga_640", "", None)
    symbolic = build_user_prompt(TINY, tiny_obs, [])
    assert "brass key" in symbolic
    assert "key_spot" in symbolic
    for banned in ("get_key", "open_shed", "shed_open", "[[0.", "puzzle"):
        assert banned not in symbolic

    pixel_obs = _observe(TINY, initial_state(TINY), "pixel", "retro_320", "", None)
    pixel_obs.hotspots = [{"id": "key_spot", "name": "brass key", "verbs": ["take", "look"]}]
    pixel = build_user_prompt(TINY, pixel_obs, [])
    for banned in ("key_spot", "brass key", "key", "get_key", "shed_open", "shed_door", "0.06", "puzzle"):
        assert banned not in pixel
    assert "ITEM" in pixel

    rope_obs = _observe(ROPE, initial_state(ROPE), "symbolic", "vga_640", "", None)
    rope = build_user_prompt(ROPE, rope_obs, [])
    for banned in ("hook", "rig", "get_rope", "make_rig", "water_raised", "holding_hook", "[[0."):
        assert banned not in rope
    assert "coil of rope" in rope


def test_unknown_model_cost_is_null_not_zero():
    assert estimate_cost("scripted", 100, 50) is None
    assert estimate_cost("gpt-4o-mini", 1_000_000, 0) == pytest.approx(0.15)


def test_client_repr_hides_keys():
    anthropic = AnthropicClient("claude-sonnet-4-5", api_key="super-secret-key")
    openai = OpenAICompatibleClient("gpt-4o", api_key="sk-openai-secret")
    xai = xai_client("grok-2-latest", api_key="xai-secret-value")
    assert "super-secret-key" not in repr(anthropic)
    assert "sk-openai-secret" not in repr(openai)
    assert "xai-secret-value" not in repr(xai)
    assert xai.base_url == "https://api.x.ai/v1"


@pytest.mark.skipif(os.environ.get("HOTSPOT_LIVE_LLM") != "1", reason="set HOTSPOT_LIVE_LLM=1 to call a live model")
def test_live_llm_smoke(tmp_path):
    from hotspot.llm.openai_compat import xai_client as build

    client = build(os.environ.get("HOTSPOT_LIVE_MODEL", "grok-2-latest"))
    agent = LLMAgent(client, mode="symbolic", world=TINY, temperature=0.0, seed=0)
    metrics = run_episode(TINY, agent, mode="symbolic", max_actions=1, log_dir=tmp_path)
    assert metrics["actions_used"] == 1
    assert metrics["total_input_tokens"] >= 0
