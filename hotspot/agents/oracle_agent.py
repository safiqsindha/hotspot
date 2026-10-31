"""Replay the solver's optimal path."""

from __future__ import annotations

from hotspot.engine import iter_hotspots
from hotspot.geom import centroid
from hotspot.schema import World
from hotspot.solver import solve


class OracleAgent:
    def __init__(self, world: World):
        self.world = world
        result = solve(world)
        if not result.solvable:
            raise RuntimeError(f"{world.id} is not solvable; oracle has no path")
        self._actions = [dict(action) for action in result.optimal_path]
        self._index = 0

    def act(self, observation) -> dict:
        if self._index >= len(self._actions):
            return {"verb": "look", "x": 0.5, "y": 0.5}
        action = dict(self._actions[self._index])
        self._index += 1
        if getattr(observation, "mode", "symbolic") == "pixel":
            return _as_pixel(self.world, action)
        return action


def _as_pixel(world: World, action: dict) -> dict:
    target = action.get("target")
    if not target:
        return dict(action)
    for scene in world.scenes:
        for hotspot in iter_hotspots(scene):
            if hotspot.id != target:
                continue
            x, y = centroid(hotspot.polygon)
            pixel = {"verb": action["verb"], "x": x, "y": y}
            if "item" in action:
                pixel["item"] = action["item"]
            return pixel
    return dict(action)
