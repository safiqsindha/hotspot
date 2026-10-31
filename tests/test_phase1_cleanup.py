"""Phase 1 cleanup: one requirements check, coverage classes, zoom budget, maps_to."""

import json
from pathlib import Path

from hotspot.engine import iter_hotspots, scene_by_id, step
from hotspot.loader import DanglingReferenceError, load_world, parse_world
from hotspot.schema import VERBS
from hotspot.solver import legal_actions
from hotspot.state import initial_state
from hotspot.validate import validate
from hotspot.harness import run_episode

ROOT = Path(__file__).resolve().parents[1]


def _world(name: str):
    return load_world(ROOT / "worlds" / name)


def _key(action: dict) -> tuple:
    return (action.get("verb"), action.get("target"), action.get("item"))


def _reachable(world):
    start = initial_state(world)
    seen = {start}
    queue = [start]
    while queue:
        state = queue.pop()
        for action in legal_actions(world, state):
            new_state, _obs, info = step(world, state, action)
            if info["state_changed"] and new_state not in seen:
                seen.add(new_state)
                queue.append(new_state)
    return seen


def _candidates(world, state):
    scene = scene_by_id(world, state.scene)
    if scene is None:
        return []
    actions = []
    extras = list(state.inventory) + ["__missing__"]
    for hotspot in iter_hotspots(scene):
        for verb in VERBS:
            spec = hotspot.verbs.get(verb)
            if spec is None:
                continue
            if spec.with_item is not None:
                actions.append({"verb": verb, "target": hotspot.id, "item": spec.with_item})
                for item in extras:
                    if item != spec.with_item:
                        actions.append({"verb": verb, "target": hotspot.id, "item": item})
            else:
                actions.append({"verb": verb, "target": hotspot.id})
    return actions


def test_legal_actions_agree_with_step_on_both_worlds():
    for name in ("tiny_key_door.json", "rope_and_well.json"):
        world = _world(name)
        for state in _reachable(world):
            allowed = {_key(action) for action in legal_actions(world, state)}
            for action in _candidates(world, state):
                _new, _obs, info = step(world, state, action)
                assert info["state_changed"] == (_key(action) in allowed), (name, state.to_dict(), action)


class _Scripted:
    def __init__(self, actions):
        self.actions = list(actions)
        self.i = 0

    def act(self, observation):
        action = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return action


def test_coverage_classes_on_rope_and_well(tmp_path):
    world = _world("rope_and_well.json")
    nowhere = {"verb": "look", "x": 0.01, "y": 0.01}
    cases = {
        "never_looked": [nowhere, nowhere],
        "looked_but_missed": [{"verb": "zoom", "box": [0.02, 0.60, 0.26, 0.92]}, nowhere],
        "found_but_misused": [{"verb": "look", "target": "rope_spot"}, nowhere],
        "used_correctly": [{"verb": "take", "target": "rope_spot"}],
    }
    for expected, actions in cases.items():
        metrics = run_episode(
            world,
            _Scripted(actions),
            mode="symbolic",
            tier="retro_320",
            max_actions=2,
            log_dir=tmp_path,
        )
        assert metrics["coverage"]["rope_spot"]["class"] == expected
        summary = metrics["coverage_summary"]
        assert summary[expected] >= 1
        assert sum(summary.values()) == len(metrics["coverage"])


def test_zoom_does_not_spend_the_action_budget(tmp_path):
    world = _world("tiny_key_door.json")
    zoom = {"verb": "zoom", "box": [0.1, 0.1, 0.4, 0.4]}
    look = {"verb": "look", "x": 0.01, "y": 0.01}
    metrics = run_episode(world, _Scripted([zoom, look]), mode="pixel", tier="retro_320", max_actions=1, log_dir=tmp_path)
    assert metrics["actions_used"] == 1
    assert metrics["wasted_actions"] == 1
    assert metrics["zoom_count"] == 1
    assert metrics["zoom_budget"] == 15


def test_sixteenth_zoom_is_refused(tmp_path):
    world = _world("tiny_key_door.json")
    zoom = {"verb": "zoom", "box": [0.1, 0.1, 0.4, 0.4]}
    look = {"verb": "look", "x": 0.01, "y": 0.01}
    actions = [zoom] * 16 + [look]
    metrics = run_episode(
        world,
        _Scripted(actions),
        mode="pixel",
        tier="retro_320",
        max_actions=1,
        zoom_budget=15,
        log_dir=tmp_path,
    )
    assert metrics["zoom_count"] == 15
    assert metrics["actions_used"] == 1
    assert metrics["wasted_actions"] == 1
    rows = [json.loads(line) for line in Path(metrics["log_path"]).read_text(encoding="utf-8").splitlines()]
    refused = [row for row in rows if row["zoom_refused"]]
    assert len(refused) == 1
    assert refused[0]["text"] == "Nothing happens."
    assert refused[0]["zoom_count"] == 15
    assert refused[0]["zoom_budget"] == 15


def test_shipped_worlds_have_no_issues_at_all():
    for name in ("tiny_key_door.json", "rope_and_well.json"):
        issues = validate(_world(name))
        assert issues == [], issues


def test_dangling_maps_to_fails():
    data = json.loads((ROOT / "worlds" / "tiny_key_door.json").read_text(encoding="utf-8"))
    data["puzzle_graph"]["nodes"][0]["maps_to"] = {"item": "no_such_item"}
    try:
        parse_world(data)
    except DanglingReferenceError as exc:
        assert any(issue.code == "dangling_maps_to" for issue in exc.issues)
    else:
        raise AssertionError("dangling maps_to should fail to load")


def test_legacy_mapping_warns_but_loads():
    data = json.loads((ROOT / "worlds" / "tiny_key_door.json").read_text(encoding="utf-8"))
    for node in data["puzzle_graph"]["nodes"]:
        node.pop("maps_to", None)
    world = parse_world(data)
    issues = validate(world)
    warnings = [issue for issue in issues if issue.code == "legacy_node_mapping"]
    assert warnings and all(issue.level == "warning" for issue in warnings)
    assert not any(issue.level == "error" for issue in issues)
