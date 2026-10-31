"""Difficulty dials: pure, deterministic, and still solvable."""

import pytest

from hotspot.dials import PRESETS, DialError, apply_dials
from hotspot.loader import load_world
from hotspot.solver import solve
from hotspot.validate import validate

TINY = load_world("worlds/tiny_key_door.json")
ROPE = load_world("worlds/rope_and_well.json")


def _dump(world):
    return world.model_dump(mode="json")


def test_apply_dials_does_not_mutate_input():
    before = _dump(ROPE)
    apply_dials(ROPE, 1, **PRESETS["medium"])
    assert _dump(ROPE) == before
    with pytest.raises(DialError):
        apply_dials(ROPE, 1, target_scale=0.3)
    assert _dump(ROPE) == before


def test_same_seed_is_byte_identical():
    first = apply_dials(ROPE, 7, **PRESETS["hard"])
    second = apply_dials(ROPE, 7, **PRESETS["hard"])
    assert first.model_dump_json() == second.model_dump_json()
    other = apply_dials(ROPE, 8, **PRESETS["hard"])
    assert first.model_dump_json() != other.model_dump_json()


@pytest.mark.parametrize("world", [TINY, ROPE], ids=["tiny", "rope"])
def test_scale_half_validates_and_keeps_optimal_length(world):
    base = solve(world).optimal_length
    dialled = apply_dials(world, 1, target_scale=0.5)
    errors = [issue for issue in validate(dialled) if issue.level == "error"]
    assert errors == []
    assert solve(dialled).optimal_length == base


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_keep_rope_optimal_length(name):
    base = solve(ROPE).optimal_length
    dialled = apply_dials(ROPE, 1, **PRESETS[name])
    assert solve(dialled).optimal_length == base
    assert not [issue for issue in validate(dialled) if issue.level == "error"]


def test_illegal_dial_raises():
    with pytest.raises(DialError) as out_of_range:
        apply_dials(ROPE, 1, target_scale=2.0)
    assert out_of_range.value.issues == []
    with pytest.raises(DialError) as too_small:
        apply_dials(ROPE, 1, target_scale=0.3)
    assert too_small.value.issues
    assert any(issue.code == "polygon_area" for issue in too_small.value.issues)
    with pytest.raises(DialError):
        apply_dials(TINY, 1, color_similarity=1.5)
