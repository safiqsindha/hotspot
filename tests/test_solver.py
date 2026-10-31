"""Solver lengths and the dead-end check."""

import json
from pathlib import Path

from hotspot.loader import load_world
from hotspot.schema import World
from hotspot.solver import solve
from hotspot.validate import validate

ROOT = Path(__file__).resolve().parents[1]


def test_optimal_lengths():
    tiny = solve(load_world(ROOT / "worlds" / "tiny_key_door.json"))
    rope = solve(load_world(ROOT / "worlds" / "rope_and_well.json"))
    assert tiny.solvable is True
    assert tiny.exhausted is False
    assert tiny.optimal_length == 2
    assert tiny.optimal_path == [
        {"verb": "take", "target": "key_spot"},
        {"verb": "use", "target": "shed_door", "item": "key"},
    ]
    assert rope.solvable is True
    assert rope.exhausted is False
    assert rope.optimal_length == 7, rope.optimal_path
    assert rope.optimal_path[-1] == {"verb": "use", "target": "well", "item": "rig"}
    assert all(step.get("target") != "coin_spot" for step in rope.optimal_path)
    assert all(step.get("item") != "coin" for step in rope.optimal_path)


def test_key_removed_is_unsolvable():
    data = json.loads((ROOT / "worlds" / "tiny_key_door.json").read_text(encoding="utf-8"))
    data["scenes"][0]["hotspots"][0]["verbs"]["take"]["effects"] = [{"hide_layer": "key_layer"}]
    world = World.model_validate(data)
    result = solve(world)
    assert result.solvable is False
    assert result.optimal_path == []
    assert result.exhausted is False


def test_dead_end_is_flagged():
    data = json.loads((ROOT / "worlds" / "tiny_key_door.json").read_text(encoding="utf-8"))
    data["scenes"][0]["hotspots"].append(
        {
            "id": "drain",
            "name": "storm drain",
            "polygon": [[0.30, 0.82], [0.48, 0.82], [0.48, 0.95], [0.30, 0.95]],
            "verbs": {
                "look": {"text": "A drain that would swallow a key."},
                "use": {
                    "with_item": "key",
                    "effects": [{"remove_item": "key"}, {"set_flag": "key_lost"}],
                },
            },
        }
    )
    world = World.model_validate(data)
    issues = [issue for issue in validate(world) if issue.level == "error"]
    dead = [issue for issue in issues if issue.code == "dead_end"]
    assert dead, issues
    assert dead[0].action["verb"] == "use"
    assert dead[0].action["target"] == "drain"
    assert dead[0].state is not None
    assert "key" in dead[0].state["inventory"]


def test_shipped_worlds_are_publishable():
    for name in ("tiny_key_door.json", "rope_and_well.json"):
        issues = [issue for issue in validate(load_world(ROOT / "worlds" / name)) if issue.level == "error"]
        assert issues == [], issues
