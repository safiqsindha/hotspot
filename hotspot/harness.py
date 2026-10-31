"""Episode runner. Logs every step and stops agents at an action cap."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from hotspot.engine import describe, iter_hotspots, scene_by_id, step
from hotspot.geom import box_covers_polygon, point_in_polygon
from hotspot.loader import load_world
from hotspot.render import TIERS, render_tier, render_zoom
from hotspot.schema import World
from hotspot.solver import solve
from hotspot.state import State, initial_state
from hotspot.validate import nodes_satisfied


@dataclass
class Observation:
    mode: str
    scene: str
    text: str
    inventory: list[str]
    hotspots: list[dict]
    goal_reached: bool
    image: Image.Image | None = None
    tier: str = "vga_640"


@dataclass
class _Coverage:
    clicked: bool = False
    zoomed: bool = False
    used_correctly: bool = False

    def to_dict(self) -> dict:
        return {
            "clicked": self.clicked,
            "zoomed": self.zoomed,
            "used_correctly": self.used_correctly,
        }


def run_episode(
    world: World,
    agent,
    mode: str = "symbolic",
    tier: str = "vga_640",
    max_actions: int | None = None,
    log_dir: str | Path = "runs",
) -> dict:
    if mode not in ("symbolic", "pixel"):
        raise ValueError("mode must be 'symbolic' or 'pixel'")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")

    result = solve(world)
    optimal = result.optimal_length if result.solvable else 0
    if max_actions is None:
        max_actions = max(20, 3 * optimal) if result.solvable else 20

    required = []
    seen_targets = set()
    for action in result.optimal_path:
        target = action.get("target")
        if target and target not in seen_targets:
            seen_targets.add(target)
            required.append(target)
    coverage = {hotspot_id: _Coverage() for hotspot_id in required}

    state = initial_state(world)
    sticky = set(nodes_satisfied(world, state))
    text = ""
    zoom_box = None
    zoom_boxes: list[list[float]] = []
    actions_used = 0
    wasted = 0
    completed = world.goal.flag in state.flags

    log_path = _log_path(world, agent, mode, log_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        while actions_used < max_actions and not completed:
            observation = _observe(world, state, mode, tier, text, zoom_box)
            action = agent.act(observation)
            if not isinstance(action, dict):
                action = {"verb": "look"}
            actions_used += 1
            new_state, engine_obs, info = step(world, state, action)
            _update_coverage(world, state, action, info, coverage)
            if info.get("zoom_box"):
                zoom_boxes.append(info["zoom_box"])
            if not info["state_changed"]:
                wasted += 1
            sticky |= nodes_satisfied(world, new_state)
            completed = bool(info["goal_reached"])
            record = {
                "step": actions_used,
                "action": action,
                "hit_hotspot": info["hit_hotspot"],
                "state_changed": info["state_changed"],
                "goal_reached": info["goal_reached"],
                "zoom_box": info.get("zoom_box"),
                "zoom_boxes": list(zoom_boxes),
                "coverage": {key: value.to_dict() for key, value in coverage.items()},
                "text": engine_obs["text"],
            }
            handle.write(json.dumps(record) + "\n")
            text = engine_obs["text"]
            zoom_box = info.get("zoom_box")
            state = new_state

    node_total = len(world.puzzle_graph.nodes)
    graph_progress = (len(sticky) / node_total) if node_total else 1.0
    efficiency = (optimal / actions_used) if actions_used else 0.0
    return {
        "world": world.id,
        "agent": type(agent).__name__,
        "mode": mode,
        "tier": tier,
        "completed": completed,
        "actions_used": actions_used,
        "optimal_length": optimal,
        "efficiency": efficiency,
        "wasted_actions": wasted,
        "graph_progress": graph_progress,
        "max_actions": max_actions,
        "coverage": {key: value.to_dict() for key, value in coverage.items()},
        "log_path": str(log_path),
    }


def _observe(world: World, state: State, mode: str, tier: str, text: str, zoom_box) -> Observation:
    body = describe(world, state, text)
    image = None
    hotspots = body["hotspots"]
    if mode == "pixel":
        hotspots = []
        if zoom_box:
            image = render_zoom(world, state, tier, zoom_box)
        else:
            image = render_tier(world, state, tier, debug=False)
    return Observation(
        mode=mode,
        scene=body["scene"],
        text=body["text"],
        inventory=list(body["inventory"]),
        hotspots=hotspots,
        goal_reached=body["goal_reached"],
        image=image,
        tier=tier,
    )


def _update_coverage(world: World, state: State, action: dict, info: dict, coverage: dict[str, _Coverage]) -> None:
    scene = scene_by_id(world, state.scene)
    hotspots = list(iter_hotspots(scene)) if scene is not None else []
    target = action.get("target")
    if target in coverage:
        coverage[target].clicked = True
    if "x" in action and "y" in action:
        x = float(action["x"])
        y = float(action["y"])
        for hotspot in hotspots:
            slot = coverage.get(hotspot.id)
            if slot is not None and point_in_polygon(x, y, hotspot.polygon):
                slot.clicked = True
    box = info.get("zoom_box")
    if box:
        for hotspot in hotspots:
            slot = coverage.get(hotspot.id)
            if slot is not None and box_covers_polygon(box, hotspot.polygon):
                slot.zoomed = True
    hit = info.get("hit_hotspot")
    if hit in coverage and info.get("state_changed"):
        coverage[hit].used_correctly = True


def _log_path(world: World, agent, mode: str, log_dir: str | Path) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%S") + f"-{time.time_ns()}"
    name = f"{world.id}-{type(agent).__name__}-{mode}-{stamp}.jsonl"
    return Path(log_dir) / name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one Hotspot episode")
    parser.add_argument("--world", required=True)
    parser.add_argument("--agent", choices=("oracle", "random"), default="oracle")
    parser.add_argument("--mode", choices=("symbolic", "pixel"), default="symbolic")
    parser.add_argument("--tier", choices=tuple(TIERS), default="vga_640")
    parser.add_argument("--max-actions", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    from hotspot.agents.oracle_agent import OracleAgent
    from hotspot.agents.random_agent import RandomAgent

    world = load_world(args.world)
    if args.agent == "oracle":
        agent = OracleAgent(world)
    else:
        agent = RandomAgent(seed=args.seed)
    metrics = run_episode(
        world,
        agent,
        mode=args.mode,
        tier=args.tier,
        max_actions=args.max_actions,
    )
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
