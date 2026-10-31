"""Pixel hit testing."""

from pathlib import Path

from hotspot.engine import step
from hotspot.geom import centroid
from hotspot.loader import load_world
from hotspot.state import initial_state

ROOT = Path(__file__).resolve().parents[1]


def _hotspot(world, hotspot_id):
    for scene in world.scenes:
        for hotspot in list(scene.hotspots) + list(scene.exits):
            if hotspot.id == hotspot_id:
                return hotspot
    raise KeyError(hotspot_id)


def test_pixel_click_at_centre_resolves_hotspot():
    world = load_world(ROOT / "worlds" / "tiny_key_door.json")
    state = initial_state(world)
    door = _hotspot(world, "shed_door")
    x, y = centroid(door.polygon)
    _state, _obs, info = step(world, state, {"verb": "look", "x": x, "y": y})
    assert info["hit_hotspot"] == "shed_door"
    assert info["state_changed"] is False
    assert set(info) >= {"hit_hotspot", "state_changed", "goal_reached"}

    _state, _obs, info = step(world, state, {"verb": "look", "x": 0.72, "y": 0.6})
    assert info["hit_hotspot"] == "shed_door"


def test_click_outside_hits_nothing():
    world = load_world(ROOT / "worlds" / "tiny_key_door.json")
    state = initial_state(world)
    new_state, obs, info = step(world, state, {"verb": "take", "x": 0.01, "y": 0.01})
    assert info["hit_hotspot"] is None
    assert info["state_changed"] is False
    assert info["goal_reached"] is False
    assert new_state == state
    assert obs["text"] == "Nothing happens."


def test_hidden_layer_is_unclickable():
    world = load_world(ROOT / "worlds" / "tiny_key_door.json")
    state = initial_state(world)
    state, _obs, info = step(world, state, {"verb": "take", "target": "key_spot"})
    assert info["hit_hotspot"] == "key_spot"
    assert info["state_changed"] is True
    assert "key" in state.inventory
    assert "key_layer" not in state.visible_layers

    key = _hotspot(world, "key_spot")
    x, y = centroid(key.polygon)
    new_state, _obs, info = step(world, state, {"verb": "take", "x": x, "y": y})
    assert info["hit_hotspot"] is None
    assert info["state_changed"] is False
    assert new_state == state
