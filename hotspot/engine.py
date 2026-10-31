"""Apply one action to a world state."""

from __future__ import annotations

from hotspot.geom import point_in_polygon
from hotspot.schema import VERBS, Hotspot, Scene, VerbSpec, World
from hotspot.state import State

NOTHING = "Nothing happens."


def scene_by_id(world: World, scene_id: str) -> Scene | None:
    for scene in world.scenes:
        if scene.id == scene_id:
            return scene
    return None


def iter_hotspots(scene: Scene):
    yield from scene.hotspots
    yield from scene.exits


def hotspot_active(hotspot: Hotspot, state: State) -> bool:
    required = hotspot.requires_layer_visible
    return required is None or required in state.visible_layers


def describe(world: World, state: State, text: str = "") -> dict:
    scene = scene_by_id(world, state.scene)
    hotspots = []
    if scene is not None:
        for hotspot in iter_hotspots(scene):
            if not hotspot_active(hotspot, state):
                continue
            hotspots.append(
                {
                    "id": hotspot.id,
                    "name": hotspot.name,
                    "verbs": [verb for verb in VERBS if verb in hotspot.verbs],
                }
            )
    return {
        "text": text,
        "scene": state.scene,
        "inventory": sorted(state.inventory),
        "hotspots": hotspots,
        "goal_reached": world.goal.flag in state.flags,
    }


def step(world: World, state: State, action: dict) -> tuple[State, dict, dict]:
    """Return ``(new_state, observation, info)``. Invalid actions are not errors."""
    verb = action.get("verb")
    if verb == "zoom":
        return _zoom(world, state, action)

    hotspot = _resolve(world, state, action)
    goal_now = world.goal.flag in state.flags
    if hotspot is None or verb not in VERBS:
        info = {"hit_hotspot": None, "state_changed": False, "goal_reached": goal_now}
        return state, describe(world, state, NOTHING), info

    spec = hotspot.verbs.get(verb)
    if spec is None or not _verb_allowed(spec, state, action):
        info = {
            "hit_hotspot": hotspot.id,
            "state_changed": False,
            "goal_reached": goal_now,
        }
        return state, describe(world, state, NOTHING), info

    new_state, texts = _apply_effects(state, spec.effects)
    text = _observation_text(verb, spec, texts, new_state != state)
    changed = new_state != state
    info = {
        "hit_hotspot": hotspot.id,
        "state_changed": changed,
        "goal_reached": world.goal.flag in new_state.flags,
    }
    return new_state, describe(world, new_state, text), info


def _zoom(world: World, state: State, action: dict) -> tuple[State, dict, dict]:
    box = action.get("box")
    goal_now = world.goal.flag in state.flags
    if not _valid_box(box):
        info = {"hit_hotspot": None, "state_changed": False, "goal_reached": goal_now}
        return state, describe(world, state, NOTHING), info
    info = {
        "hit_hotspot": None,
        "state_changed": False,
        "goal_reached": goal_now,
        "zoom_box": [float(v) for v in box],
    }
    return state, describe(world, state, "You look closer."), info


def _resolve(world: World, state: State, action: dict) -> Hotspot | None:
    if "target" in action and action.get("target") is not None:
        return _resolve_target(world, state, str(action["target"]))
    if "x" in action and "y" in action:
        return _resolve_point(world, state, float(action["x"]), float(action["y"]))
    return None


def _resolve_target(world: World, state: State, target: str) -> Hotspot | None:
    scene = scene_by_id(world, state.scene)
    if scene is None:
        return None
    for hotspot in iter_hotspots(scene):
        if hotspot.id == target and hotspot_active(hotspot, state):
            return hotspot
    return None


def _resolve_point(world: World, state: State, x: float, y: float) -> Hotspot | None:
    scene = scene_by_id(world, state.scene)
    if scene is None:
        return None
    hit = None
    for hotspot in iter_hotspots(scene):
        if not hotspot_active(hotspot, state):
            continue
        if point_in_polygon(x, y, hotspot.polygon):
            hit = hotspot
    return hit


def _verb_allowed(spec: VerbSpec, state: State, action: dict) -> bool:
    if spec.with_item is not None and action.get("item") != spec.with_item:
        return False
    return _requires_ok(spec, state)


def _requires_ok(spec: VerbSpec, state: State) -> bool:
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


def _apply_effects(state: State, effects) -> tuple[State, list[str]]:
    inventory = set(state.inventory)
    flags = set(state.flags)
    layers = set(state.visible_layers)
    scene = state.scene
    texts: list[str] = []
    for effect in effects:
        if effect.set_flag:
            flags.add(effect.set_flag)
        if effect.clear_flag:
            flags.discard(effect.clear_flag)
        if effect.give_item:
            inventory.add(effect.give_item)
        if effect.remove_item:
            inventory.discard(effect.remove_item)
        if effect.show_layer:
            layers.add(effect.show_layer)
        if effect.hide_layer:
            layers.discard(effect.hide_layer)
        if effect.goto_scene:
            scene = effect.goto_scene
        if effect.text:
            texts.append(effect.text)
    new_state = State(
        scene=scene,
        inventory=frozenset(inventory),
        flags=frozenset(flags),
        visible_layers=frozenset(layers),
    )
    return new_state, texts


def _observation_text(verb: str, spec: VerbSpec, texts: list[str], changed: bool) -> str:
    if texts:
        return " ".join(texts)
    if spec.text:
        return spec.text
    if not changed:
        return NOTHING
    if verb == "take":
        return "Taken."
    if verb == "go":
        return "You head that way."
    return "Done."


def _valid_box(box) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        values = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    return all(0.0 <= v <= 1.0 for v in values)
