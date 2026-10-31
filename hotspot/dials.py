"""Difficulty dials. Pure: same world, seed and arguments always rebuild the same world."""

from __future__ import annotations

import random
from pathlib import Path

from hotspot.geom import box_covers_polygon, centroid
from hotspot.schema import World
from hotspot.solver import solve
from hotspot.validate import Issue, validate

PALETTE = [
    "#ff5555",
    "#55ff55",
    "#5555ff",
    "#ffff55",
    "#ff55ff",
    "#55ffff",
    "#aaaaaa",
    "#ffffff",
    "#aa5500",
    "#00aa00",
]

PRESETS = {
    "easy": {"target_scale": 1.1, "decoy_shapes": 1, "decoy_hotspots": 0, "color_similarity": 0.0},
    "medium": {"target_scale": 0.9, "decoy_shapes": 3, "decoy_hotspots": 2, "color_similarity": 0.35},
    "hard": {"target_scale": 0.6, "decoy_shapes": 5, "decoy_hotspots": 3, "color_similarity": 0.85},
}


class DialError(Exception):
    def __init__(self, message: str, issues: list[Issue]):
        self.issues = issues
        super().__init__(message)


def apply_dials(
    world: World,
    seed: int,
    *,
    target_scale: float = 1.0,
    decoy_shapes: int = 0,
    decoy_hotspots: int = 0,
    color_similarity: float = 0.0,
) -> World:
    if not 0.3 <= target_scale <= 1.5:
        raise DialError(f"target_scale {target_scale} is outside 0.3–1.5", [])
    if not 0.0 <= color_similarity <= 1.0:
        raise DialError(f"color_similarity {color_similarity} is outside 0–1", [])
    original = world.model_dump(mode="json")
    data = json_copy(original)
    _scale_targets(data, target_scale)
    rng = random.Random(seed)
    colors = _target_colors(data)
    scenes = _scene_order(data)
    forbidden = {scene["id"]: _scene_polygons(scene) for scene in scenes}
    _scatter(scenes, forbidden, rng, decoy_shapes, "shape", colors, color_similarity)
    _scatter(scenes, forbidden, rng, decoy_hotspots, "hotspot", colors, color_similarity)
    dialled = World.model_validate(data)
    issues = [issue for issue in validate(dialled) if issue.level == "error"]
    if issues:
        raise DialError("dialled world failed validate()", issues)
    if world.model_dump(mode="json") != original:
        raise DialError("apply_dials mutated the input world", [])
    return dialled


def render_preset_screens(world: World, out_dir: str | Path = "docs/dials", seed: int = 1) -> list[Path]:
    from hotspot.render import render_tier
    from hotspot.state import initial_state

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for name, dials in PRESETS.items():
        dialled = apply_dials(world, seed, **dials)
        image = render_tier(dialled, initial_state(dialled), "retro_320")
        path = destination / f"{world.id}_{name}_retro_320.png"
        image.save(path, optimize=True)
        written.append(path)
    return written


def json_copy(value):
    import json

    return json.loads(json.dumps(value))


def _scene_order(data: dict) -> list[dict]:
    scenes = list(data["scenes"])
    scenes.sort(key=lambda scene: (scene["id"] != data["start_scene"], scene["id"]))
    return scenes


def _scatter(scenes, forbidden, rng, count, kind, colors, similarity) -> None:
    if count <= 0:
        return
    cursor = 0
    for index in range(count):
        placed = False
        for offset in range(len(scenes)):
            scene = scenes[(cursor + offset) % len(scenes)]
            box = _place_box(rng, forbidden[scene["id"]])
            if box is None:
                continue
            color = _decoy_color(rng, colors, similarity)
            if kind == "shape":
                scene["shapes"].append(
                    {"kind": "rect", "box": box, "color": color, "label": f"decoy-shape-{index}"}
                )
            else:
                scene["shapes"].append(
                    {"kind": "rect", "box": box, "color": color, "label": f"decoy-rock-{index}"}
                )
                scene["hotspots"].append(
                    {
                        "id": f"decoy_spot_{index}",
                        "name": "rock",
                        "polygon": _poly(box),
                        "verbs": {"look": {"text": "Just a rock."}},
                    }
                )
            forbidden[scene["id"]].append(_poly(box))
            cursor = (cursor + offset + 1) % len(scenes)
            placed = True
            break
        if not placed:
            raise DialError("could not place a decoy without overlapping a hotspot", [])


def _scale_targets(data: dict, scale: float) -> None:
    if scale == 1.0:
        return
    linked: dict[str, list] = {}
    for scene in data["scenes"]:
        for layer in scene["item_layers"]:
            linked.setdefault(layer["id"], []).append(layer["shape"])
        for hotspot in list(scene["hotspots"]) + list(scene.get("exits") or []):
            layer_id = hotspot.get("requires_layer_visible")
            if layer_id:
                linked.setdefault(layer_id, []).append(hotspot)
    for shapes in linked.values():
        for shape in shapes:
            if "polygon" in shape:
                shape["polygon"] = _scale_polygon(shape["polygon"], scale)
            elif "box" in shape and shape["box"] is not None:
                shape["box"] = _scale_box(shape["box"], scale)
            elif "shape" in shape:
                inner = shape["shape"]
                if inner.get("box") is not None:
                    inner["box"] = _scale_box(inner["box"], scale)
                if inner.get("points"):
                    inner["points"] = _scale_polygon(inner["points"], scale)


def _scale_polygon(polygon, scale: float):
    cx, cy = centroid(polygon)
    return [[cx + (x - cx) * scale, cy + (y - cy) * scale] for x, y in polygon]


def _scale_box(box, scale: float):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2
    cy = (y0 + y1) / 2
    return [
        cx + (x0 - cx) * scale,
        cy + (y0 - cy) * scale,
        cx + (x1 - cx) * scale,
        cy + (y1 - cy) * scale,
    ]


def _scene_polygons(scene: dict) -> list:
    polygons = []
    for hotspot in list(scene["hotspots"]) + list(scene.get("exits") or []):
        polygons.append(hotspot["polygon"])
    return polygons


def _target_colors(data: dict) -> list[str]:
    colors = []
    for scene in data["scenes"]:
        for layer in scene["item_layers"]:
            if layer.get("decoy"):
                continue
            color = layer["shape"]["color"]
            if color not in colors:
                colors.append(color)
    return colors or ["#ffd000"]


def _decoy_color(rng: random.Random, targets: list[str], similarity: float) -> str:
    base = _parse(rng.choice(PALETTE))
    target = _parse(rng.choice(targets))
    mixed = tuple(int(round(base[i] + (target[i] - base[i]) * similarity)) for i in range(3))
    return "#{:02x}{:02x}{:02x}".format(*mixed)


def _parse(value: str) -> tuple[int, int, int]:
    text = value.lstrip("#")
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def _place_box(rng: random.Random, forbidden: list) -> list[float] | None:
    widths = (0.10, 0.08, 0.07)
    heights = (0.12, 0.09, 0.08)
    candidates = []
    for width, height in zip(widths, heights):
        step = 0.05
        x = 0.02
        while x + width <= 0.98:
            y = 0.02
            while y + height <= 0.98:
                candidates.append([round(x, 4), round(y, 4), round(x + width, 4), round(y + height, 4)])
                y += step
            x += step
    rng.shuffle(candidates)
    for box in candidates:
        polygon = _poly(box)
        if any(box_covers_polygon(box, other) or box_covers_polygon(_bounds(other), polygon) for other in forbidden):
            continue
        return box
    return None


def _poly(box: list[float]) -> list[list[float]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _bounds(polygon) -> list[float]:
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def main() -> None:
    from hotspot.loader import load_world

    world = load_world("worlds/rope_and_well.json")
    base = solve(world).optimal_length
    for name, dials in PRESETS.items():
        dialled = apply_dials(world, 1, **dials)
        print(f"{name} optimal {solve(dialled).optimal_length} (base {base})")
    for path in render_preset_screens(world):
        print(path)


if __name__ == "__main__":
    main()
