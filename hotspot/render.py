"""Procedural renderer. Hotspots are invisible unless debug overlay is on."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from hotspot.engine import hotspot_active, iter_hotspots, scene_by_id
from hotspot.loader import load_world
from hotspot.schema import Shape, World
from hotspot.state import State, initial_state

TIERS = {
    "retro_320": (320, 200, 16),
    "vga_640": (640, 480, 256),
    "hd_1280": (1280, 720, None),
    "uhd_3840": (3840, 2160, None),
}


def render(world: World, state: State, width: int, height: int, debug: bool = False) -> Image.Image:
    """Draw the scene at ``width``×``height``.

    Geometry is drawn at an integer multiple of the target and nearest-neighbour
    downscaled so tiers share the same normalized layout. Palette quantization
    is applied by :func:`render_tier`, not here.
    """
    image = _paint(world, state, width, height)
    if debug:
        _overlay(image, world, state)
    return image


def render_tier(world: World, state: State, tier: str, debug: bool = False) -> Image.Image:
    if tier not in TIERS:
        known = ", ".join(TIERS)
        raise KeyError(f"unknown tier {tier!r}; expected one of {known}")
    width, height, colors = TIERS[tier]
    scale = _render_scale(width, height)
    master = _paint(world, state, width * scale, height * scale)
    image = master.resize((width, height), Image.Resampling.NEAREST) if scale != 1 else master
    image = _quantize(image, colors)
    if debug:
        _overlay(image, world, state)
    return image


def render_zoom(world: World, state: State, tier: str, box: list[float]) -> Image.Image:
    """Crop ``box`` out of a higher-resolution paint and scale it to the tier size."""
    width, height, colors = TIERS[tier]
    scale = max(_render_scale(width, height), 2)
    master = _paint(world, state, width * scale, height * scale)
    x0, y0, x1, y1 = (float(v) for v in box)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    left = int(round(x0 * master.width))
    top = int(round(y0 * master.height))
    right = int(round(x1 * master.width))
    bottom = int(round(y1 * master.height))
    right = min(master.width, max(right, left + 1))
    bottom = min(master.height, max(bottom, top + 1))
    left = max(0, min(left, master.width - 1))
    top = max(0, min(top, master.height - 1))
    crop = master.crop((left, top, right, bottom))
    image = crop.resize((width, height), Image.Resampling.NEAREST)
    return _quantize(image, colors)


def write_screens(worlds_dir: str | Path = "worlds", out_dir: str | Path = "docs/screens") -> list[Path]:
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for path in sorted(Path(worlds_dir).glob("*.json")):
        world = load_world(path)
        state = initial_state(world)
        for tier in TIERS:
            plain = render_tier(world, state, tier, debug=False)
            debug = render_tier(world, state, tier, debug=True)
            plain_path = destination / f"{world.id}_{tier}.png"
            debug_path = destination / f"{world.id}_{tier}_debug.png"
            plain.save(plain_path, optimize=True)
            debug.save(debug_path, optimize=True)
            written.extend((plain_path, debug_path))
    return written


def _render_scale(width: int, height: int) -> int:
    scale = 4
    while scale > 1 and (width * scale) * (height * scale) > 6_000_000:
        scale //= 2
    return scale


def _paint(world: World, state: State, width: int, height: int) -> Image.Image:
    scene = scene_by_id(world, state.scene)
    color = _parse_color(scene.background.color) if scene is not None else (0, 0, 0)
    image = Image.new("RGB", (width, height), color)
    if scene is None:
        return image
    draw = ImageDraw.Draw(image)
    for shape in scene.shapes:
        _draw_shape(draw, shape, width, height)
    for layer in scene.item_layers:
        if layer.id in state.visible_layers:
            _draw_shape(draw, layer.shape, width, height)
    return image


def _draw_shape(draw: ImageDraw.ImageDraw, shape: Shape, width: int, height: int) -> None:
    color = _parse_color(shape.color)
    if shape.kind in ("rect", "ellipse"):
        assert shape.box is not None
        box = _box_px(shape.box, width, height)
        if shape.kind == "rect":
            draw.rectangle(box, fill=color)
        else:
            draw.ellipse(box, fill=color)
    elif shape.kind == "polygon" and shape.points:
        points = [(point[0] * width, point[1] * height) for point in shape.points]
        draw.polygon(points, fill=color)


def _overlay(image: Image.Image, world: World, state: State) -> None:
    scene = scene_by_id(world, state.scene)
    if scene is None:
        return
    draw = ImageDraw.Draw(image)
    width, height = image.size
    size = max(12, height // 22)
    try:
        font = ImageFont.load_default(size=size)
    except TypeError:
        font = ImageFont.load_default()
    width_px = max(1, height // 180)
    for hotspot in iter_hotspots(scene):
        points = [(point[0] * width, point[1] * height) for point in hotspot.polygon]
        active = hotspot_active(hotspot, state)
        color = (255, 230, 40) if active else (150, 150, 150)
        draw.line(points + [points[0]], fill=color, width=width_px)
        cx = sum(point[0] for point in points) / len(points)
        cy = sum(point[1] for point in points) / len(points)
        label = hotspot.name
        draw.text((cx + 1, cy + 1), label, fill=(0, 0, 0), font=font)
        draw.text((cx, cy), label, fill=(255, 255, 255), font=font)


def _quantize(image: Image.Image, colors: int | None) -> Image.Image:
    image = image.convert("RGB")
    if not colors:
        return image
    found = image.getcolors(maxcolors=colors)
    if found is not None:
        return image
    # Most frequent colors first; ties break on the RGB tuple so the result is stable.
    ranked = image.getcolors(maxcolors=image.width * image.height) or []
    ranked.sort(key=lambda item: (-item[0], item[1]))
    palette = []
    for _count, rgb in ranked:
        if rgb not in palette:
            palette.append(rgb)
        if len(palette) == colors:
            break
    pixels = list(image.getdata())
    mapped = [_nearest(pixel, palette) for pixel in pixels]
    out = Image.new("RGB", image.size)
    out.putdata(mapped)
    return out


def _nearest(rgb: tuple[int, int, int], palette: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    best = palette[0]
    best_distance = 1 << 62
    red, green, blue = rgb
    for candidate in palette:
        distance = (red - candidate[0]) ** 2 + (green - candidate[1]) ** 2 + (blue - candidate[2]) ** 2
        if distance < best_distance:
            best_distance = distance
            best = candidate
    return best


def _box_px(box: list[float], width: int, height: int) -> list[float]:
    x0, y0, x1, y1 = box
    return [x0 * width, y0 * height, x1 * width, y1 * height]


def _parse_color(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(channel * 2 for channel in text)
    return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)


def main() -> None:
    written = write_screens()
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
