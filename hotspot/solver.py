"""Breadth-first search over symbolic actions."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from hotspot.engine import hotspot_active, iter_hotspots, scene_by_id, step
from hotspot.schema import VERBS, World
from hotspot.state import State, initial_state


@dataclass
class SolveResult:
    solvable: bool
    optimal_path: list[dict] = field(default_factory=list)
    optimal_length: int = 0
    states_explored: int = 0
    exhausted: bool = False


def solve(world: World, max_states: int = 100_000) -> SolveResult:
    start = initial_state(world)
    if world.goal.flag in start.flags:
        return SolveResult(True, [], 0, 1, False)

    parents: dict[State, tuple[State, dict] | None] = {start: None}
    queue: deque[State] = deque([start])
    expanded = 0
    while queue:
        if expanded >= max_states:
            return SolveResult(False, [], 0, len(parents), True)
        state = queue.popleft()
        expanded += 1
        for action in legal_actions(world, state):
            new_state, _obs, info = step(world, state, action)
            if not info["state_changed"] or new_state in parents:
                continue
            parents[new_state] = (state, dict(action))
            if info["goal_reached"]:
                path = _reconstruct(parents, new_state)
                return SolveResult(True, path, len(path), len(parents), False)
            queue.append(new_state)
    return SolveResult(False, [], 0, len(parents), False)


def legal_actions(world: World, state: State) -> list[dict]:
    """State-changing symbolic actions whose requirements are already met."""
    scene = scene_by_id(world, state.scene)
    if scene is None:
        return []
    actions = []
    for hotspot in iter_hotspots(scene):
        if not hotspot_active(hotspot, state):
            continue
        for verb in VERBS:
            spec = hotspot.verbs.get(verb)
            if spec is None:
                continue
            action = {"verb": verb, "target": hotspot.id}
            if spec.with_item is not None:
                if spec.with_item not in state.inventory:
                    continue
                action["item"] = spec.with_item
            if not _requirements_met(spec, state):
                continue
            _new_state, _obs, info = step(world, state, action)
            if info["state_changed"]:
                actions.append(action)
    return actions


def _requirements_met(spec, state: State) -> bool:
    requires = spec.requires
    if requires is None:
        return True
    if requires.flag is not None and requires.flag not in state.flags:
        return False
    if requires.not_flag is not None and requires.not_flag in state.flags:
        return False
    if requires.has_item is not None and requires.has_item not in state.inventory:
        return False
    if requires.layer_visible is not None and requires.layer_visible not in state.visible_layers:
        return False
    return True


def _reconstruct(parents: dict, state: State) -> list[dict]:
    path = []
    cursor = state
    while parents[cursor] is not None:
        previous, action = parents[cursor]
        path.append(dict(action))
        cursor = previous
    path.reverse()
    return path
