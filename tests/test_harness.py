"""Oracle efficiency, the action cap, and JSONL coverage."""

import json
from pathlib import Path

import pytest

from hotspot.agents.oracle_agent import OracleAgent
from hotspot.agents.random_agent import RandomAgent
from hotspot.harness import run_episode
from hotspot.loader import load_world

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("world_name", ["tiny_key_door.json", "rope_and_well.json"])
@pytest.mark.parametrize("mode", ["symbolic", "pixel"])
def test_oracle_completes_with_efficiency_one(world_name, mode, tmp_path):
    world = load_world(ROOT / "worlds" / world_name)
    metrics = run_episode(
        world,
        OracleAgent(world),
        mode=mode,
        tier="vga_640",
        log_dir=tmp_path,
    )
    assert metrics["completed"] is True
    assert metrics["actions_used"] == metrics["optimal_length"]
    assert metrics["efficiency"] == 1.0
    assert metrics["wasted_actions"] == 0
    assert metrics["graph_progress"] == 1.0
    assert metrics["coverage"]
    for hotspot_id, coverage in metrics["coverage"].items():
        assert coverage["clicked"] is True, hotspot_id
        assert coverage["used_correctly"] is True, hotspot_id


def test_random_agent_respects_action_cap(tmp_path):
    world = load_world(ROOT / "worlds" / "rope_and_well.json")
    metrics = run_episode(
        world,
        RandomAgent(seed=1),
        mode="symbolic",
        max_actions=5,
        log_dir=tmp_path,
    )
    assert metrics["actions_used"] <= 5
    assert metrics["max_actions"] == 5
    if not metrics["completed"]:
        assert metrics["actions_used"] == 5

    default = run_episode(
        world,
        RandomAgent(seed=2),
        mode="pixel",
        tier="retro_320",
        log_dir=tmp_path,
    )
    cap = max(20, 3 * default["optimal_length"])
    assert default["actions_used"] <= cap
    assert default["max_actions"] == cap


def test_jsonl_contains_coverage_fields(tmp_path):
    world = load_world(ROOT / "worlds" / "tiny_key_door.json")
    metrics = run_episode(
        world,
        OracleAgent(world),
        mode="pixel",
        tier="retro_320",
        log_dir=tmp_path,
    )
    lines = Path(metrics["log_path"]).read_text(encoding="utf-8").splitlines()
    assert len(lines) == metrics["actions_used"] == 2
    for line in lines:
        row = json.loads(line)
        assert "hit_hotspot" in row
        assert "state_changed" in row
        assert "zoom_box" in row
        assert "zoom_boxes" in row
        assert "coverage" in row
        sample = next(iter(row["coverage"].values()))
        assert {"clicked", "zoomed", "used_correctly"} <= set(sample)
