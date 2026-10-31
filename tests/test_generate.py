"""Generated worlds are deterministic, valid, and oracle-perfect."""

import pytest

from hotspot.agents.oracle_agent import OracleAgent
from hotspot.generate import TEMPLATES, build_world, generate_worlds, is_duplicate
from hotspot.harness import run_episode
from hotspot.validate import validate


@pytest.mark.parametrize("template", TEMPLATES)
def test_each_template_validates_and_oracle_is_perfect(template, tmp_path):
    world = build_world(template, 7, index=1)
    issues = validate(world)
    assert [issue for issue in issues if issue.level == "error"] == []
    assert [issue for issue in issues if issue.level == "warning"] == []
    for mode in ("symbolic", "pixel"):
        metrics = run_episode(
            world,
            OracleAgent(world),
            mode=mode,
            tier="retro_320",
            log_dir=tmp_path / mode,
        )
        assert metrics["completed"] is True
        assert metrics["efficiency"] == 1.0
        assert 2 <= metrics["optimal_length"] <= 12


def test_same_seed_generates_the_same_worlds():
    first = generate_worlds(4, 1)
    second = generate_worlds(4, 1)
    assert [item.world.model_dump(mode="json") for item in first] == [
        item.world.model_dump(mode="json") for item in second
    ]
    assert [item.template for item in first] == [item.template for item in second]
    assert len({item.world.id for item in first}) == 4


def test_duplicate_signatures_are_rejected():
    world = build_world("key_door", 5, index=0)
    seen: set[str] = set()
    assert is_duplicate("key_door", world, seen) is False
    assert is_duplicate("key_door", world, seen) is True
    other = build_world("key_door", 9, index=1)
    assert is_duplicate("key_door", other, seen) is False
