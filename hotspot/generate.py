"""Seeded world generator. Six templates, no overlapping required hotspots.

A red herring is an item layer with ``decoy: true``. ``validate`` still requires
every other given item to be used, and skips decoy items.

CLI: ``python -m hotspot.generate --count 8 --seed 1 --out worlds/generated``
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from hotspot.geom import box_covers_polygon
from hotspot.schema import World
from hotspot.solver import solve
from hotspot.validate import validate

TEMPLATES = (
    "key_door",
    "tool_obstacle",
    "two_scene_fetch",
    "combine",
    "chain3",
    "red_herring",
)

# Large, mutually disjoint slots. At retro_320 a slot is about 90×70 px.
_SLOTS = {
    "nw": [0.04, 0.08, 0.32, 0.42],
    "ne": [0.68, 0.08, 0.96, 0.42],
    "sw": [0.04, 0.56, 0.32, 0.92],
    "se": [0.68, 0.56, 0.96, 0.92],
    "c": [0.38, 0.30, 0.62, 0.62],
}

# (id, display name, color). Colors stay far from the dark backgrounds.
THINGS = (
    ("key", "brass key", "#ffd000"),
    ("gem", "blue gem", "#00e5ff"),
    ("crowbar", "iron crowbar", "#f2f2f2"),
    ("torch", "oil torch", "#ff5a1f"),
    ("lens", "glass lens", "#7dff6a"),
    ("bell", "brass bell", "#ff4fd8"),
    ("coin", "gold coin", "#ffe14a"),
    ("map", "folded map", "#f4e1b5"),
    ("hook", "iron hook", "#d0d8e6"),
    ("rope", "coil of rope", "#ff8a3d"),
    ("flask", "green flask", "#3dff7a"),
    ("gear", "brass gear", "#ffcf70"),
)

_BACKGROUNDS = ("#142033", "#1a1a2e", "#10241c", "#241810")
_GROUND = "#2c3e50"
_DOOR = "#7ec8ff"
_EXIT = "#f4efe6"
_OBSTACLE = "#ff2d55"
_BENCH = "#e0a060"
_DEVICE = "#c9a0ff"


@dataclass
class Generated:
    world: World
    template: str
    seed: int
    optimal_length: int


def build_world(template: str, seed: int, *, index: int = 0) -> World:
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}")
    rng = random.Random(seed)
    slots = _layout(seed)
    background = _BACKGROUNDS[seed % len(_BACKGROUNDS)]
    if template == "key_door":
        data = _key_door(rng.sample(THINGS, 1)[0], slots, background, index)
    elif template == "tool_obstacle":
        data = _tool_obstacle(rng.sample(THINGS, 1)[0], slots, background, index)
    elif template == "two_scene_fetch":
        data = _two_scene_fetch(rng.sample(THINGS, 1)[0], slots, background, index)
    elif template == "combine":
        first, second, third = rng.sample(THINGS, 3)
        data = _combine(first, second, third, slots, background, index)
    elif template == "chain3":
        tool, prize = rng.sample(THINGS, 2)
        data = _chain3(tool, prize, slots, background, index)
    else:
        useful, decoy = rng.sample(THINGS, 2)
        data = _red_herring(useful, decoy, slots, background, index)
    world = World.model_validate(data)
    _require_disjoint(world)
    return world


def structural_signature(template: str, world: World) -> str:
    items = []
    layout = []
    for scene in world.scenes:
        for layer in scene.item_layers:
            if layer.item:
                items.append(layer.item)
        for hotspot in list(scene.hotspots) + list(scene.exits):
            points = ",".join(f"{x:.3f}:{y:.3f}" for x, y in hotspot.polygon)
            layout.append(f"{scene.id}:{hotspot.id}:{points}")
    return template + "|" + ",".join(sorted(items)) + "|" + ";".join(layout)


def is_duplicate(template: str, world: World, seen: set[str]) -> bool:
    """Record ``world``'s signature. Return True if it was already seen."""
    signature = structural_signature(template, world)
    if signature in seen:
        return True
    seen.add(signature)
    return False


def generate_worlds(count: int, seed: int) -> list[Generated]:
    if count < 1:
        raise ValueError("count must be at least 1")
    rng = random.Random(seed)
    seen: set[str] = set()
    kept: list[Generated] = []
    attempts = 0
    limit = max(40, count * 25)
    while len(kept) < count:
        attempts += 1
        if attempts > limit:
            raise RuntimeError(
                f"accepted {len(kept)} of {count} worlds "
                f"({attempts - 1} candidates tried; duplicates or invalid worlds rejected)"
            )
        template = TEMPLATES[rng.randrange(len(TEMPLATES))]
        world_seed = rng.randrange(0, 2**31)
        world = build_world(template, world_seed, index=len(kept))
        errors = [issue for issue in validate(world) if issue.level == "error"]
        if errors:
            continue
        result = solve(world)
        if not result.solvable or not 2 <= result.optimal_length <= 12:
            continue
        if is_duplicate(template, world, seen):
            continue
        dumped = world.model_dump(mode="json")
        dumped["id"] = f"{template}_{len(kept):02d}"
        world = World.model_validate(dumped)
        kept.append(Generated(world, template, world_seed, result.optimal_length))
    return kept


def write_worlds(generated: list[Generated], out_dir: str | Path, *, batch_seed: int) -> Path:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for stale in destination.glob("*.json"):
        stale.unlink()
    rows = []
    for item in generated:
        filename = f"{item.world.id}.json"
        payload = item.world.model_dump(mode="json")
        (destination / filename).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        rows.append(
            {
                "id": item.world.id,
                "template": item.template,
                "seed": item.seed,
                "optimal_length": item.optimal_length,
                "scenes": len(item.world.scenes),
                "dials": None,
                "file": filename,
            }
        )
    manifest = destination / "manifest.json"
    body = {"seed": batch_seed, "count": len(generated), "worlds": rows}
    manifest.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")
    return manifest


def _layout(seed: int) -> dict[str, list[float]]:
    variant = seed % 3
    return {name: _mirror(box, variant) for name, box in _SLOTS.items()}


def _mirror(box: list[float], variant: int) -> list[float]:
    x0, y0, x1, y1 = box
    if variant == 1:
        return [round(1 - x1, 4), y0, round(1 - x0, 4), y1]
    if variant == 2:
        return [x0, round(1 - y1, 4), x1, round(1 - y0, 4)]
    return list(box)


def _key_door(thing, slots, background, index) -> dict:
    item_id, name, _color = thing
    flag = "door_open"
    scene = _scene(
        "yard",
        background,
        shapes=[_shape(slots["se"], _DOOR, "door")],
        layers=[_layer(thing, slots["sw"])],
        hotspots=[
            _take(thing, slots["sw"]),
            _use_item(
                "door",
                "locked door",
                slots["se"],
                item_id,
                flag,
                f"The {name} turns and the door opens.",
            ),
        ],
    )
    return _world(
        index,
        "key_door",
        f"Open the door with the {name}.",
        f"Use the {name} on the door.",
        "yard",
        flag,
        [scene],
        [
            {"id": f"get_{item_id}", "kind": "action", "maps_to": {"item": item_id}},
            {"id": "opened", "kind": "goal", "maps_to": {"flag": flag}},
        ],
        [[f"get_{item_id}", "opened"]],
    )


def _tool_obstacle(thing, slots, background, index) -> dict:
    item_id, name, _color = thing
    flag = "path_clear"
    yard = _scene(
        "yard",
        background,
        shapes=[_shape(slots["ne"], _EXIT, "passage")],
        layers=[_layer(thing, slots["sw"])],
        hotspots=[_take(thing, slots["sw"])],
        exits=[_go("to_site", "narrow passage", slots["ne"], "site", "You squeeze through the passage.")],
    )
    site = _scene(
        "site",
        "#1c140c",
        shapes=[_shape(slots["c"], _OBSTACLE, "rubble"), _shape(slots["se"], _EXIT, "way back")],
        layers=[],
        hotspots=[
            _use_item(
                "rubble",
                "fallen rubble",
                slots["c"],
                item_id,
                flag,
                f"You lever the rubble aside with the {name}.",
                remove=True,
            )
        ],
        exits=[_go("to_yard", "way back", slots["se"], "yard", "You climb back out.")],
    )
    return _world(
        index,
        "tool_obstacle",
        f"Clear the rubble with the {name}.",
        f"Take the {name} and clear the rubble.",
        "yard",
        flag,
        [yard, site],
        [
            {"id": f"get_{item_id}", "kind": "action", "maps_to": {"item": item_id}},
            {"id": "cleared", "kind": "goal", "maps_to": {"flag": flag}},
        ],
        [[f"get_{item_id}", "cleared"]],
    )


def _two_scene_fetch(thing, slots, background, index) -> dict:
    item_id, name, _color = thing
    flag = "altar_lit"
    hall = _scene(
        "hall",
        background,
        shapes=[_shape(slots["se"], _DEVICE, "altar"), _shape(slots["ne"], _EXIT, "arch")],
        layers=[],
        hotspots=[
            _use_item(
                "altar",
                "stone altar",
                slots["se"],
                item_id,
                flag,
                f"You set the {name} on the altar.",
            )
        ],
        exits=[_go("to_grove", "garden arch", slots["ne"], "grove", "You walk into the grove.")],
    )
    grove = _scene(
        "grove",
        "#0e2418",
        shapes=[_shape(slots["ne"], _EXIT, "arch")],
        layers=[_layer(thing, slots["sw"])],
        hotspots=[_take(thing, slots["sw"])],
        exits=[_go("to_hall", "arch to the hall", slots["ne"], "hall", "You return to the hall.")],
    )
    return _world(
        index,
        "two_scene_fetch",
        f"Fetch the {name}.",
        f"Bring the {name} back to the altar.",
        "hall",
        flag,
        [hall, grove],
        [
            {"id": f"get_{item_id}", "kind": "action", "maps_to": {"item": item_id}},
            {"id": "offered", "kind": "goal", "maps_to": {"flag": flag}},
        ],
        [[f"get_{item_id}", "offered"]],
    )


def _combine(first, second, third, slots, background, index) -> dict:
    a_id, a_name, _a = first
    b_id, b_name, _b = second
    c_id, c_name, _c = third
    flag = "device_on"
    scene = _scene(
        "workshop",
        background,
        shapes=[_shape(slots["c"], _BENCH, "bench"), _shape(slots["se"], _DEVICE, "device")],
        layers=[_layer(first, slots["nw"]), _layer(second, slots["ne"])],
        hotspots=[
            _take(first, slots["nw"]),
            _take(second, slots["ne"]),
            {
                "id": "bench",
                "name": "workbench",
                "polygon": _poly(slots["c"]),
                "verbs": {
                    "look": {"text": f"A bench. The {a_name} and the {b_name} might fit together."},
                    "use": {
                        "with_item": a_id,
                        "requires": {"has_item": b_id},
                        "effects": [
                            {
                                "remove_item": a_id,
                                "text": f"You combine them into a {c_name}.",
                            },
                            {"remove_item": b_id},
                            {"give_item": c_id},
                        ],
                    },
                },
            },
            _use_item(
                "device",
                "strange device",
                slots["se"],
                c_id,
                flag,
                f"The {c_name} clicks into the device.",
            ),
        ],
    )
    return _world(
        index,
        "combine",
        f"Make a {c_name}.",
        f"Combine the {a_name} and the {b_name}, then use the {c_name}.",
        "workshop",
        flag,
        [scene],
        [
            {"id": f"get_{a_id}", "kind": "action", "maps_to": {"item": a_id}},
            {"id": f"get_{b_id}", "kind": "action", "maps_to": {"item": b_id}},
            {"id": f"make_{c_id}", "kind": "action", "maps_to": {"item": c_id}},
            {"id": "powered", "kind": "goal", "maps_to": {"flag": flag}},
        ],
        [
            [f"get_{a_id}", f"make_{c_id}"],
            [f"get_{b_id}", f"make_{c_id}"],
            [f"make_{c_id}", "powered"],
        ],
    )


def _chain3(tool, prize, slots, background, index) -> dict:
    tool_id, tool_name, _tool_color = tool
    prize_id, prize_name, _prize_color = prize
    flag = "gate_open"
    yard = _scene(
        "yard",
        background,
        shapes=[_shape(slots["se"], _DOOR, "gate"), _shape(slots["ne"], _EXIT, "cave mouth")],
        layers=[],
        hotspots=[
            _use_item(
                "gate",
                "locked gate",
                slots["se"],
                prize_id,
                flag,
                f"The {prize_name} opens the gate.",
            )
        ],
        exits=[_go("to_cave", "cave mouth", slots["ne"], "cave", "You step into the cave.")],
    )
    cave = _scene(
        "cave",
        "#16120e",
        shapes=[_shape(slots["c"], _OBSTACLE, "crate"), _shape(slots["ne"], _EXIT, "daylight")],
        layers=[_layer(tool, slots["sw"])],
        hotspots=[
            _take(tool, slots["sw"]),
            {
                "id": "crate",
                "name": "nailed crate",
                "polygon": _poly(slots["c"]),
                "verbs": {
                    "look": {"text": f"A nailed crate. Something inside glints like a {prize_name}."},
                    "use": {
                        "with_item": tool_id,
                        "effects": [
                            {
                                "remove_item": tool_id,
                                "give_item": prize_id,
                                "text": f"You pry the crate with the {tool_name} and find a {prize_name}.",
                            }
                        ],
                    },
                },
            },
        ],
        exits=[_go("to_yard", "way back", slots["ne"], "yard", "You climb back to the yard.")],
    )
    return _world(
        index,
        "chain3",
        f"Open the gate with the {prize_name}.",
        f"Find a {tool_name}, get the {prize_name}, and open the gate.",
        "yard",
        flag,
        [yard, cave],
        [
            {"id": f"get_{tool_id}", "kind": "action", "maps_to": {"item": tool_id}},
            {"id": f"get_{prize_id}", "kind": "action", "maps_to": {"item": prize_id}},
            {"id": "opened", "kind": "goal", "maps_to": {"flag": flag}},
        ],
        [[f"get_{tool_id}", f"get_{prize_id}"], [f"get_{prize_id}", "opened"]],
    )


def _red_herring(useful, decoy, slots, background, index) -> dict:
    data = _key_door(useful, slots, background, index)
    item_id, name, _color = decoy
    scene = data["scenes"][0]
    scene["item_layers"].append(_layer(decoy, slots["nw"], decoy=True))
    scene["hotspots"].append(_take(decoy, slots["nw"]))
    data["id"] = f"red_herring_{index:02d}"
    data["title"] = f"Ignore the {name}."
    data["goal"]["description"] = f"Use the {useful[1]} on the door. The {name} is useless."
    return data


def _world(index, template, title, description, start, flag, scenes, nodes, edges) -> dict:
    return {
        "id": f"{template}_{index:02d}",
        "title": title,
        "start_scene": start,
        "goal": {"flag": flag, "description": description},
        "scenes": scenes,
        "puzzle_graph": {"nodes": nodes, "edges": edges},
    }


def _scene(scene_id, background, *, shapes, layers, hotspots, exits=None) -> dict:
    body = [_shape([0.0, 0.50, 1.0, 1.0], _GROUND, "ground"), *shapes]
    return {
        "id": scene_id,
        "background": {"color": background},
        "shapes": body,
        "item_layers": layers,
        "hotspots": hotspots,
        "exits": exits or [],
    }


def _shape(box, color, label) -> dict:
    return {"kind": "rect", "box": list(box), "color": color, "label": label}


def _layer(thing, box, decoy: bool = False) -> dict:
    item_id, _name, color = thing
    layer = {
        "id": f"{item_id}_layer",
        "item": item_id,
        "visible": True,
        "shape": {"kind": "rect", "box": list(box), "color": color},
    }
    if decoy:
        layer["decoy"] = True
    return layer


def _take(thing, box) -> dict:
    item_id, name, _color = thing
    return {
        "id": f"{item_id}_spot",
        "name": name,
        "polygon": _poly(box),
        "requires_layer_visible": f"{item_id}_layer",
        "verbs": {
            "look": {"text": f"A {name} lying in plain sight."},
            "take": {
                "effects": [
                    {
                        "give_item": item_id,
                        "hide_layer": f"{item_id}_layer",
                        "text": f"You take the {name}.",
                    }
                ]
            },
        },
    }


def _use_item(hotspot_id, name, box, item_id, flag, text, remove: bool = True) -> dict:
    effect = {"set_flag": flag, "text": text}
    if remove:
        effect["remove_item"] = item_id
    return {
        "id": hotspot_id,
        "name": name,
        "polygon": _poly(box),
        "verbs": {
            "look": {"text": f"You look at the {name}."},
            "use": {
                "with_item": item_id,
                "requires": {"not_flag": flag},
                "effects": [effect],
            },
        },
    }


def _go(hotspot_id, name, box, scene_id, text) -> dict:
    return {
        "id": hotspot_id,
        "name": name,
        "polygon": _poly(box),
        "verbs": {
            "look": {"text": f"You could go through the {name}."},
            "go": {"effects": [{"goto_scene": scene_id, "text": text}]},
        },
    }


def _poly(box) -> list[list[float]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _bounds(polygon) -> list[float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def _require_disjoint(world: World) -> None:
    for scene in world.scenes:
        hotspots = list(scene.hotspots) + list(scene.exits)
        for index, hotspot in enumerate(hotspots):
            box = _bounds(hotspot.polygon)
            for other in hotspots[index + 1 :]:
                if box_covers_polygon(box, other.polygon):
                    raise RuntimeError(f"{world.id}: {hotspot.id} overlaps {other.id} in {scene.id}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate solvable Hotspot worlds")
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="worlds/generated")
    args = parser.parse_args(argv)
    generated = generate_worlds(args.count, args.seed)
    manifest = write_worlds(generated, args.out, batch_seed=args.seed)
    print(f"accepted {len(generated)} worlds -> {manifest}")
    for item in generated:
        print(f"  {item.world.id} template={item.template} length={item.optimal_length} scenes={len(item.world.scenes)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
