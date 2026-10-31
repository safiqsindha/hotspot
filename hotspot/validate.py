"""Structural checks. A world is publishable when this returns no errors."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

from hotspot.engine import hotspot_active, iter_hotspots, scene_by_id, step
from hotspot.geom import polygon_area
from hotspot.schema import World
from hotspot.solver import legal_actions, solve
from hotspot.state import State, initial_state


@dataclass
class Issue:
    level: str
    code: str
    message: str
    state: dict | None = None
    action: dict | None = None


def validate(world: World, min_area: float = 0.002, max_states: int = 100_000) -> list[Issue]:
    issues = []
    issues.extend(reference_issues(world))
    issues.extend(polygon_issues(world, min_area))
    issues.extend(graph_structure_issues(world))
    issues.extend(gameplay_issues(world, max_states))
    return issues


def reference_issues(world: World) -> list[Issue]:
    issues: list[Issue] = []
    scenes: dict[str, object] = {}
    layers: set[str] = set()
    hotspot_ids: set[str] = set()
    for scene in world.scenes:
        if scene.id in scenes:
            issues.append(_err("duplicate_id", f"duplicate scene id {scene.id!r}"))
        scenes[scene.id] = scene
        for layer in scene.item_layers:
            if layer.id in layers:
                issues.append(_err("duplicate_id", f"duplicate layer id {layer.id!r}"))
            layers.add(layer.id)
        for hotspot in iter_hotspots(scene):
            if hotspot.id in hotspot_ids:
                issues.append(_err("duplicate_id", f"duplicate hotspot id {hotspot.id!r}"))
            hotspot_ids.add(hotspot.id)

    if world.start_scene not in scenes:
        issues.append(_err("missing_scene", f"start_scene {world.start_scene!r} does not exist"))

    given, flags = _defined_items_and_flags(world)
    if world.goal.flag not in flags:
        issues.append(_err("missing_flag", f"goal flag {world.goal.flag!r} is never set"))

    for scene in world.scenes:
        for hotspot in iter_hotspots(scene):
            layer_id = hotspot.requires_layer_visible
            if layer_id is not None and layer_id not in layers:
                issues.append(
                    _err(
                        "missing_layer",
                        f"{hotspot.id} requires missing layer {layer_id!r}",
                    )
                )
            for verb, spec in hotspot.verbs.items():
                if spec.with_item is not None and spec.with_item not in given:
                    issues.append(
                        _err(
                            "missing_item",
                            f"{hotspot.id}.{verb} with_item {spec.with_item!r} is never given",
                        )
                    )
                requires = spec.requires
                if requires is not None:
                    if requires.flag is not None and requires.flag not in flags:
                        issues.append(
                            _err(
                                "missing_flag",
                                f"{hotspot.id}.{verb} requires missing flag {requires.flag!r}",
                            )
                        )
                    if requires.not_flag is not None and requires.not_flag not in flags:
                        issues.append(
                            _err(
                                "missing_flag",
                                f"{hotspot.id}.{verb} requires missing flag {requires.not_flag!r}",
                            )
                        )
                    if requires.has_item is not None and requires.has_item not in given:
                        issues.append(
                            _err(
                                "missing_item",
                                f"{hotspot.id}.{verb} requires missing item {requires.has_item!r}",
                            )
                        )
                    if requires.layer_visible is not None and requires.layer_visible not in layers:
                        issues.append(
                            _err(
                                "missing_layer",
                                f"{hotspot.id}.{verb} requires missing layer {requires.layer_visible!r}",
                            )
                        )
                for effect in spec.effects:
                    if effect.goto_scene is not None and effect.goto_scene not in scenes:
                        issues.append(
                            _err(
                                "missing_scene",
                                f"{hotspot.id}.{verb} goes to missing scene {effect.goto_scene!r}",
                            )
                        )
                    for layer_name, label in (
                        (effect.show_layer, "show_layer"),
                        (effect.hide_layer, "hide_layer"),
                    ):
                        if layer_name is not None and layer_name not in layers:
                            issues.append(
                                _err(
                                    "missing_layer",
                                    f"{hotspot.id}.{verb} {label} {layer_name!r} does not exist",
                                )
                            )
                    if effect.remove_item is not None and effect.remove_item not in given:
                        issues.append(
                            _err(
                                "missing_item",
                                f"{hotspot.id}.{verb} removes {effect.remove_item!r}, which is never given",
                            )
                        )
    return issues


def polygon_issues(world: World, min_area: float) -> list[Issue]:
    issues = []
    for scene in world.scenes:
        for hotspot in iter_hotspots(scene):
            for x, y in hotspot.polygon:
                if x < 0.0 or x > 1.0 or y < 0.0 or y > 1.0:
                    issues.append(
                        _err(
                            "polygon_bounds",
                            f"{hotspot.id} has a point outside 0–1: [{x}, {y}]",
                        )
                    )
                    break
            area = polygon_area(hotspot.polygon)
            if area < min_area:
                issues.append(
                    _err(
                        "polygon_area",
                        f"{hotspot.id} area {area:.4f} is below {min_area}",
                    )
                )
    return issues


def graph_structure_issues(world: World) -> list[Issue]:
    issues = []
    nodes = world.puzzle_graph.nodes
    ids = [node.id for node in nodes]
    if len(ids) != len(set(ids)):
        issues.append(_err("duplicate_id", "puzzle graph has duplicate node ids"))
    known = set(ids)
    for src, dst in world.puzzle_graph.edges:
        if src not in known or dst not in known:
            issues.append(_err("missing_node", f"puzzle edge {(src, dst)} references an unknown node"))
    if _has_cycle(ids, world.puzzle_graph.edges):
        issues.append(_err("puzzle_cycle", "puzzle graph has a cycle"))
    items, flags = _defined_items_and_flags(world)
    flags.add(world.goal.flag)
    hotspots = {
        hotspot.id
        for scene in world.scenes
        for hotspot in iter_hotspots(scene)
    }
    for node in nodes:
        if not _node_maps(node.id, node.kind, items, flags, hotspots, world.goal.flag):
            issues.append(
                _err(
                    "unmapped_node",
                    f"puzzle node {node.id!r} does not map to an item, flag, or hotspot",
                )
            )
    return issues


def gameplay_issues(world: World, max_states: int) -> list[Issue]:
    issues: list[Issue] = []
    result = solve(world, max_states=max_states)
    if result.exhausted:
        issues.append(_err("search_exhausted", f"solver hit the {max_states} state cap"))

    parent, succ, order, exhausted = _reachable(world, max_states)
    if exhausted and not result.exhausted:
        issues.append(_err("search_exhausted", f"reachability search hit the {max_states} state cap"))

    start = initial_state(world)
    can = _can_reach_goal(world, parent, succ)
    if start not in can:
        issues.append(_err("goal_unreachable", f"goal flag {world.goal.flag!r} is not reachable"))
    elif not exhausted:
        for state in order:
            if state not in can:
                continue
            for action, nxt in succ.get(state, []):
                if nxt not in can:
                    issues.append(
                        Issue(
                            level="error",
                            code="dead_end",
                            message=(
                                f"action {action['verb']} {action.get('target')} from scene "
                                f"{state.scene} strands the player"
                            ),
                            state=state.to_dict(),
                            action=dict(action),
                        )
                    )
                    break
            else:
                continue
            break

    given, _flags = _defined_items_and_flags(world)
    used = _used_items(world)
    for item in sorted(given - used):
        issues.append(_err("unused_item", f"item {item!r} is given but never used"))

    reachable_ids: set[str] = set()
    for state in parent:
        scene = scene_by_id(world, state.scene)
        if scene is None:
            continue
        for hotspot in iter_hotspots(scene):
            if hotspot_active(hotspot, state):
                reachable_ids.add(hotspot.id)
    for scene in world.scenes:
        for hotspot in iter_hotspots(scene):
            if hotspot.id not in reachable_ids:
                issues.append(
                    _err("unreachable_hotspot", f"hotspot {hotspot.id!r} is never clickable")
                )

    issues.extend(_graph_order_issues(world, max_states))
    if not result.solvable and start in can:
        # Reachability and the solver disagreed; surface the solver result too.
        issues.append(_err("goal_unreachable", f"solver could not reach {world.goal.flag!r}"))
    return issues


def nodes_satisfied(world: World, state: State) -> set[str]:
    """Puzzle nodes whose item or flag is true in this state (not sticky)."""
    found = set()
    for node in world.puzzle_graph.nodes:
        if _node_true(node.id, state):
            found.add(node.id)
    return found


def _node_true(node_id: str, state: State) -> bool:
    reversed_id = "_".join(reversed(node_id.split("_")))
    if node_id in state.flags or reversed_id in state.flags:
        return True
    if node_id in state.inventory:
        return True
    if node_id.startswith("get_") and node_id[4:] in state.inventory:
        return True
    if node_id.startswith("make_") and node_id[5:] in state.inventory:
        return True
    return False


def _graph_order_issues(world: World, max_states: int) -> list[Issue]:
    edges = list(world.puzzle_graph.edges)
    if not edges:
        return []
    start = initial_state(world)
    sticky = frozenset(nodes_satisfied(world, start))
    queue = deque([(start, sticky)])
    seen = {(start, sticky)}
    reported: set[tuple[str, str]] = set()
    issues: list[Issue] = []
    expanded = 0
    while queue:
        if expanded >= max_states:
            issues.append(_err("search_exhausted", "puzzle-graph check hit the state cap"))
            break
        state, have = queue.popleft()
        expanded += 1
        for action in legal_actions(world, state):
            new_state, _obs, info = step(world, state, action)
            if not info["state_changed"]:
                continue
            gained = frozenset(nodes_satisfied(world, new_state))
            new_have = have | gained
            for src, dst in edges:
                if dst in new_have and src not in new_have and (src, dst) not in reported:
                    reported.add((src, dst))
                    issues.append(
                        _err(
                            "puzzle_order",
                            f"node {dst!r} was reached without {src!r}",
                        )
                    )
            key = (new_state, new_have)
            if key not in seen:
                seen.add(key)
                queue.append((new_state, new_have))
    return issues


def _reachable(world: World, max_states: int):
    start = initial_state(world)
    parent: dict[State, tuple[State, dict] | None] = {start: None}
    succ: dict[State, list[tuple[dict, State]]] = {start: []}
    order: list[State] = []
    queue: deque[State] = deque([start])
    exhausted = False
    while queue:
        if len(order) >= max_states:
            exhausted = True
            break
        state = queue.popleft()
        order.append(state)
        succ.setdefault(state, [])
        for action in legal_actions(world, state):
            new_state, _obs, info = step(world, state, action)
            if not info["state_changed"]:
                continue
            succ[state].append((dict(action), new_state))
            if new_state not in parent:
                parent[new_state] = (state, dict(action))
                succ.setdefault(new_state, [])
                queue.append(new_state)
    return parent, succ, order, exhausted


def _can_reach_goal(world: World, parent: dict, succ: dict) -> set[State]:
    goals = [state for state in parent if world.goal.flag in state.flags]
    can = set(goals)
    incoming: dict[State, list[State]] = defaultdict(list)
    for state, pairs in succ.items():
        for _action, nxt in pairs:
            incoming[nxt].append(state)
    queue = deque(goals)
    while queue:
        current = queue.popleft()
        for previous in incoming[current]:
            if previous not in can:
                can.add(previous)
                queue.append(previous)
    return can


def _defined_items_and_flags(world: World) -> tuple[set[str], set[str]]:
    items: set[str] = set()
    flags: set[str] = set()
    for scene in world.scenes:
        for hotspot in iter_hotspots(scene):
            for spec in hotspot.verbs.values():
                for effect in spec.effects:
                    if effect.give_item:
                        items.add(effect.give_item)
                    if effect.set_flag:
                        flags.add(effect.set_flag)
    return items, flags


def _used_items(world: World) -> set[str]:
    used: set[str] = set()
    for scene in world.scenes:
        for hotspot in iter_hotspots(scene):
            for spec in hotspot.verbs.values():
                if spec.with_item:
                    used.add(spec.with_item)
                if spec.requires is not None and spec.requires.has_item:
                    used.add(spec.requires.has_item)
                for effect in spec.effects:
                    if effect.remove_item:
                        used.add(effect.remove_item)
    return used


def _node_maps(node_id: str, kind: str, items: set[str], flags: set[str], hotspots: set[str], goal: str) -> bool:
    reversed_id = "_".join(reversed(node_id.split("_")))
    if node_id in items or node_id in flags or node_id in hotspots:
        return True
    if reversed_id in items or reversed_id in flags or reversed_id in hotspots:
        return True
    if node_id.startswith("get_") and node_id[4:] in items:
        return True
    if node_id.startswith("make_") and node_id[5:] in items:
        return True
    if kind == "goal" and (node_id == goal or reversed_id == goal):
        return True
    return False


def _has_cycle(node_ids: list[str], edges: list[tuple[str, str]]) -> bool:
    adjacent: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for src, dst in edges:
        if src in adjacent and dst in adjacent:
            adjacent[src].append(dst)
    color = {node_id: 0 for node_id in node_ids}

    def visit(node_id: str) -> bool:
        color[node_id] = 1
        for nxt in adjacent[node_id]:
            if color[nxt] == 1:
                return True
            if color[nxt] == 0 and visit(nxt):
                return True
        color[node_id] = 2
        return False

    return any(color[node_id] == 0 and visit(node_id) for node_id in node_ids)


def _err(code: str, message: str) -> Issue:
    return Issue(level="error", code=code, message=message)
