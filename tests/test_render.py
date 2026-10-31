"""Fidelity tiers and deterministic paints."""

from pathlib import Path

import pytest

from hotspot.loader import load_world
from hotspot.render import TIERS, render_tier
from hotspot.state import State, initial_state

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def tiny():
    return load_world(ROOT / "worlds" / "tiny_key_door.json")


@pytest.mark.parametrize("tier", list(TIERS))
def test_tier_size_and_palette(tiny, tier):
    state = initial_state(tiny)
    image = render_tier(tiny, state, tier)
    width, height, colors = TIERS[tier]
    assert image.size == (width, height)
    assert image.mode == "RGB"
    if colors is not None:
        found = image.getcolors(maxcolors=colors + 1)
        assert found is not None
        assert len(found) <= colors


def test_render_is_deterministic(tiny):
    state = initial_state(tiny)
    first = render_tier(tiny, state, "vga_640").tobytes()
    second = render_tier(tiny, state, "vga_640").tobytes()
    assert first == second


def test_hiding_a_layer_changes_the_image(tiny):
    state = initial_state(tiny)
    hidden = State(
        scene=state.scene,
        inventory=state.inventory,
        flags=state.flags,
        visible_layers=state.visible_layers - {"key_layer"},
    )
    before = render_tier(tiny, state, "retro_320").tobytes()
    after = render_tier(tiny, hidden, "retro_320").tobytes()
    assert before != after
