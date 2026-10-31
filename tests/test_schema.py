"""Schema and reference checks."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from hotspot.loader import DanglingReferenceError, load_world, parse_world
from hotspot.schema import World

ROOT = Path(__file__).resolve().parents[1]


def _tiny() -> dict:
    return json.loads((ROOT / "worlds" / "tiny_key_door.json").read_text(encoding="utf-8"))


def test_both_worlds_load():
    tiny = load_world(ROOT / "worlds" / "tiny_key_door.json")
    rope = load_world(ROOT / "worlds" / "rope_and_well.json")
    assert tiny.id == "tiny_key_door"
    assert tiny.title == "The Locked Shed"
    assert tiny.goal.flag == "shed_open"
    assert rope.id == "rope_and_well"
    assert len(rope.scenes) == 2
    assert {scene.id for scene in rope.scenes} == {"courtyard", "shed"}


def test_unknown_field_fails():
    data = _tiny()
    data["author"] = "nope"
    with pytest.raises(ValidationError):
        World.model_validate(data)


def test_dangling_id_fails():
    data = _tiny()
    data["scenes"][0]["hotspots"][1]["verbs"]["use"]["effects"].append({"goto_scene": "nowhere"})
    with pytest.raises(DanglingReferenceError) as caught:
        parse_world(data)
    assert any(issue.code == "missing_scene" for issue in caught.value.issues)
