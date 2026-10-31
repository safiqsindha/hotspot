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
from hotspot.llm.pricing import estimate_cost

COVERAGE_CLASSES = ("never_looked", "looked_but_missed", "found_but_misused", "used_correctly")


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
            "class": _coverage_class(self),
        }


def _coverage_class(slot: _Coverage) -> str:
    if slot.used_correctly:
        return "used_correctly"
    if slot.clicked:
        return "found_but_misused"
    if slot.zoomed:
        return "looked_but_missed"
    return "never_looked"


def run_episode(
    world: World,
    agent,
    mode: str = "symbolic",
    tier: str = "vga_640",
    max_actions: int | None = None,
    log_dir: str | Path = "runs",
    zoom_budget: int = 15,
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
    intended: dict[str, set[tuple[str, str | None]]] = {}
    for action in result.optimal_path:
        target = action.get("target")
        if not target:
            continue
        intended.setdefault(target, set()).add((action.get("verb"), action.get("item")))
        if target not in seen_targets:
            seen_targets.add(target)
            required.append(target)
    coverage = {hotspot_id: _Coverage() for hotspot_id in required}
    satisfied: dict[str, set[tuple[str, str | None]]] = {hotspot_id: set() for hotspot_id in required}

    state = initial_state(world)
    used_hotspots: set[str] = set()
    sticky = set(nodes_satisfied(world, state, used_hotspots))
    text = ""
    zoom_box = None
    zoom_boxes: list[list[float]] = []
    actions_used = 0
    wasted = 0
    zoom_count = 0
    refused_zooms = 0
    completed = world.goal.flag in state.flags
    total_input = 0
    total_output = 0
    model_name = getattr(agent, "model_name", type(agent).__name__)

    log_path = _log_path(world, agent, mode, log_dir)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # Refused zooms past this many start counting as wasted actions so a
    # zoom-only agent cannot spin forever. The first refusal is still free.
    refuse_grace = zoom_budget
    with log_path.open("w", encoding="utf-8") as handle:
        while actions_used < max_actions and not completed:
            observation = _observe(world, state, mode, tier, text, zoom_box)
            action = agent.act(observation)
            if not isinstance(action, dict):
                action = {"verb": "look"}
            meta = _step_meta(agent, model_name)
            total_input += int(meta["input_tokens"])
            total_output += int(meta["output_tokens"])

            if action.get("verb") == "zoom" and _zoom_box_ok(action.get("box")):
                if zoom_count >= zoom_budget:
                    refused_zooms += 1
                    counted = refused_zooms > refuse_grace
                    if counted:
                        actions_used += 1
                        wasted += 1
                    record = _record(
                        actions_used if counted else 0,
                        action,
                        {"hit_hotspot": None, "state_changed": False, "goal_reached": completed, "zoom_box": None},
                        "Nothing happens.",
                        coverage,
                        zoom_boxes,
                        zoom_count,
                        zoom_budget,
                        meta,
                        zoom_refused=True,
                    )
                    handle.write(json.dumps(record) + "\n")
                    text = "Nothing happens."
                    zoom_box = None
                    continue
                new_state, engine_obs, info = step(world, state, action)
                zoom_count += 1
                if info.get("zoom_box"):
                    zoom_boxes.append(info["zoom_box"])
                _update_coverage(world, state, action, info, coverage, intended, satisfied)
                record = _record(
                    actions_used,
                    action,
                    info,
                    engine_obs["text"],
                    coverage,
                    zoom_boxes,
                    zoom_count,
                    zoom_budget,
                    meta,
                    zoom_refused=False,
                )
                handle.write(json.dumps(record) + "\n")
                text = engine_obs["text"]
                zoom_box = info.get("zoom_box")
                state = new_state
                continue

            actions_used += 1
            new_state, engine_obs, info = step(world, state, action)
            _update_coverage(world, state, action, info, coverage, intended, satisfied)
            if info.get("zoom_box"):
                zoom_boxes.append(info["zoom_box"])
            if not info["state_changed"]:
                wasted += 1
            if info["state_changed"] and info.get("hit_hotspot"):
                used_hotspots.add(info["hit_hotspot"])
            sticky |= nodes_satisfied(world, new_state, used_hotspots)
            completed = bool(info["goal_reached"])
            record = _record(
                actions_used,
                action,
                info,
                engine_obs["text"],
                coverage,
                zoom_boxes,
                zoom_count,
                zoom_budget,
                meta,
                zoom_refused=False,
            )
            handle.write(json.dumps(record) + "\n")
            text = engine_obs["text"]
            zoom_box = info.get("zoom_box")
            state = new_state

    node_total = len(world.puzzle_graph.nodes)
    graph_progress = (len(sticky) / node_total) if node_total else 1.0
    efficiency = (optimal / actions_used) if actions_used else 0.0
    summary = {name: 0 for name in COVERAGE_CLASSES}
    for slot in coverage.values():
        summary[_coverage_class(slot)] += 1
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
        "zoom_count": zoom_count,
        "zoom_budget": zoom_budget,
        "coverage": {key: value.to_dict() for key, value in coverage.items()},
        "coverage_summary": summary,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "est_cost_usd": estimate_cost(model_name if total_input or total_output else model_name, total_input, total_output),
        "model": model_name,
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


def _update_coverage(world, state, action, info, coverage, intended, satisfied) -> None:
    scene = scene_by_id(world, state.scene)
    hotspots = list(iter_hotspots(scene)) if scene is not None else []
    target = action.get("target")
    verb = action.get("verb")
    item = action.get("item")
    point = None
    if "x" in action and "y" in action:
        point = (float(action["x"]), float(action["y"]))
        for hotspot in hotspots:
            slot = coverage.get(hotspot.id)
            if slot is not None and point_in_polygon(point[0], point[1], hotspot.polygon):
                slot.clicked = True
    if target in coverage:
        coverage[target].clicked = True
    box = info.get("zoom_box")
    if box:
        for hotspot in hotspots:
            slot = coverage.get(hotspot.id)
            if slot is not None and box_covers_polygon(box, hotspot.polygon):
                slot.zoomed = True
    hit = info.get("hit_hotspot")
    for hotspot_id, triples in intended.items():
        pair = (verb, item)
        if pair not in triples:
            continue
        matched = False
        if target == hotspot_id:
            matched = True
        elif hit == hotspot_id and point is not None:
            matched = True
        if matched:
            satisfied[hotspot_id].add(pair)
    for hotspot_id, slot in coverage.items():
        need = intended.get(hotspot_id, set())
        if need and need <= satisfied[hotspot_id]:
            slot.used_correctly = True


def _zoom_box_ok(box) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError):
        return False
    return all(0.0 <= value <= 1.0 for value in values)


def _step_meta(agent, model_name: str) -> dict:
    meta = getattr(agent, "last_step_meta", None) or {}
    raw = meta.get("raw_reply", "")
    if not isinstance(raw, str):
        raw = str(raw)
    return {
        "input_tokens": int(meta.get("input_tokens", 0) or 0),
        "output_tokens": int(meta.get("output_tokens", 0) or 0),
        "latency_s": float(meta.get("latency_s", 0.0) or 0.0),
        "raw_reply": raw[:2000],
        "parse_retries": int(meta.get("parse_retries", 0) or 0),
        "model": meta.get("model", model_name),
        "temperature": meta.get("temperature"),
        "seed": meta.get("seed"),
    }


def _record(step_n, action, info, text, coverage, zoom_boxes, zoom_count, zoom_budget, meta, zoom_refused: bool) -> dict:
    return {
        "step": step_n,
        "action": action,
        "hit_hotspot": info.get("hit_hotspot"),
        "state_changed": info.get("state_changed", False),
        "goal_reached": info.get("goal_reached", False),
        "zoom_box": info.get("zoom_box"),
        "zoom_boxes": list(zoom_boxes),
        "zoom_count": zoom_count,
        "zoom_budget": zoom_budget,
        "zoom_refused": zoom_refused,
        "coverage": {key: value.to_dict() for key, value in coverage.items()},
        "text": text,
        "input_tokens": meta["input_tokens"],
        "output_tokens": meta["output_tokens"],
        "latency_s": meta["latency_s"],
        "raw_reply": meta["raw_reply"],
        "parse_retries": meta["parse_retries"],
        "model": meta["model"],
        "temperature": meta["temperature"],
        "seed": meta["seed"],
    }


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
