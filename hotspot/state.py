"""Hashable game state."""

from __future__ import annotations

from dataclasses import dataclass

from hotspot.schema import World


@dataclass(frozen=True)
class State:
    scene: str
    inventory: frozenset[str]
    flags: frozenset[str]
    visible_layers: frozenset[str]

    def to_dict(self) -> dict:
        return {
            "scene": self.scene,
            "inventory": sorted(self.inventory),
            "flags": sorted(self.flags),
            "visible_layers": sorted(self.visible_layers),
        }


def initial_state(world: World) -> State:
    visible = [
        layer.id
        for scene in world.scenes
        for layer in scene.item_layers
        if layer.visible
    ]
    return State(
        scene=world.start_scene,
        inventory=frozenset(),
        flags=frozenset(),
        visible_layers=frozenset(visible),
    )
