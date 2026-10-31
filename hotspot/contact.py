"""Contact sheets of generated worlds. One image per tier, plain and debug.

Renders every scene so a required item that starts off-screen is still visible.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from hotspot.loader import load_world
from hotspot.render import render_tier
from hotspot.state import initial_state

TIERS = ("retro_320", "vga_640")


def write_contact_sheets(
    worlds_dir: str | Path = "worlds/generated",
    out_dir: str | Path = "docs/generated",
) -> list[Path]:
    paths = [path for path in sorted(Path(worlds_dir).glob("*.json")) if path.name != "manifest.json"]
    if not paths:
        raise FileNotFoundError(f"no worlds in {worlds_dir}")
    panels = []
    for path in paths:
        world = load_world(path)
        base = initial_state(world)
        for scene in world.scenes:
            panels.append((f"{world.id} / {scene.id}", world, replace(base, scene=scene.id)))
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for tier in TIERS:
        for debug in (False, True):
            sheet = _sheet(panels, tier, debug)
            name = f"{tier}_debug.png" if debug else f"{tier}.png"
            target = destination / name
            sheet.save(target, optimize=True)
            written.append(target)
    return written


def _sheet(panels, tier: str, debug: bool) -> Image.Image:
    sample = render_tier(panels[0][1], panels[0][2], tier, debug=False)
    cell_w, cell_h = sample.size
    label_h = max(16, cell_h // 12)
    pad = 8
    columns = 4 if len(panels) > 1 else 1
    rows = (len(panels) + columns - 1) // columns
    width = columns * cell_w + (columns + 1) * pad
    height = rows * (cell_h + label_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (width, height), (12, 14, 18))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=max(12, label_h - 4))
    except TypeError:
        font = ImageFont.load_default()
    for index, (label, world, state) in enumerate(panels):
        image = render_tier(world, state, tier, debug=debug)
        column = index % columns
        row = index // columns
        x = pad + column * (cell_w + pad)
        y = pad + row * (cell_h + label_h + pad)
        sheet.paste(image, (x, y))
        draw.text((x, y + cell_h + 2), label, fill=(230, 230, 230), font=font)
    return sheet


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render a contact sheet of generated worlds")
    parser.add_argument("--worlds", default="worlds/generated")
    parser.add_argument("--out", default="docs/generated")
    args = parser.parse_args(argv)
    for path in write_contact_sheets(args.worlds, args.out):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
