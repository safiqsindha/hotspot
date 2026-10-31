"""Uniform random symbolic or pixel actions. The harness enforces the cap."""

from __future__ import annotations

import random


class RandomAgent:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)

    def act(self, observation) -> dict:
        if observation.mode == "pixel":
            return self._pixel(observation)
        return self._symbolic(observation)

    def _symbolic(self, observation) -> dict:
        hotspots = list(observation.hotspots)
        if not hotspots:
            return {"verb": "look", "x": 0.5, "y": 0.5}
        hotspot = self._rng.choice(hotspots)
        verb = self._rng.choice(hotspot["verbs"])
        action = {"verb": verb, "target": hotspot["id"]}
        if verb == "use" and observation.inventory:
            action["item"] = self._rng.choice(list(observation.inventory))
        return action

    def _pixel(self, observation) -> dict:
        if self._rng.random() < 0.25:
            x0 = self._rng.random() * 0.7
            y0 = self._rng.random() * 0.7
            return {"verb": "zoom", "box": [x0, y0, min(1.0, x0 + 0.3), min(1.0, y0 + 0.3)]}
        action = {
            "verb": self._rng.choice(["look", "take", "use", "talk", "go"]),
            "x": self._rng.random(),
            "y": self._rng.random(),
        }
        if action["verb"] == "use" and observation.inventory:
            action["item"] = self._rng.choice(list(observation.inventory))
        return action
