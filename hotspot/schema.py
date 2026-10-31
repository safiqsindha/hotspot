"""Pydantic models for a Hotspot world. Unknown fields are rejected."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

VERBS = ("look", "take", "use", "talk", "go")


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Shape(_Model):
    kind: Literal["rect", "ellipse", "polygon"]
    box: list[float] | None = None
    points: list[list[float]] | None = None
    color: str
    label: str | None = None

    @model_validator(mode="after")
    def _shape_fields(self) -> Shape:
        if self.kind in ("rect", "ellipse"):
            if self.box is None or len(self.box) != 4:
                raise ValueError(f"{self.kind} shape requires box [x0, y0, x1, y1]")
        elif self.kind == "polygon":
            if not self.points or len(self.points) < 3:
                raise ValueError("polygon shape requires at least 3 points")
        return self


class Effect(_Model):
    set_flag: str | None = None
    clear_flag: str | None = None
    give_item: str | None = None
    remove_item: str | None = None
    show_layer: str | None = None
    hide_layer: str | None = None
    goto_scene: str | None = None
    text: str | None = None


class Requires(_Model):
    flag: str | None = None
    not_flag: str | None = None
    has_item: str | None = None
    layer_visible: str | None = None


class VerbSpec(_Model):
    text: str | None = None
    with_item: str | None = None
    requires: Requires | None = None
    effects: list[Effect] = Field(default_factory=list)


class Hotspot(_Model):
    id: str
    name: str
    polygon: list[list[float]]
    requires_layer_visible: str | None = None
    verbs: dict[str, VerbSpec]

    @field_validator("polygon")
    @classmethod
    def _polygon(cls, value: list[list[float]]) -> list[list[float]]:
        if len(value) < 3:
            raise ValueError("polygon needs at least 3 points")
        for point in value:
            if len(point) != 2:
                raise ValueError("polygon points must be [x, y]")
        return value

    @field_validator("verbs")
    @classmethod
    def _verbs(cls, value: dict[str, VerbSpec]) -> dict[str, VerbSpec]:
        unknown = sorted(set(value) - set(VERBS))
        if unknown:
            raise ValueError(f"unknown verbs: {unknown}")
        if not value:
            raise ValueError("hotspot needs at least one verb")
        return value


class ItemLayer(_Model):
    id: str
    item: str | None = None
    visible: bool = True
    decoy: bool = False
    shape: Shape


class Background(_Model):
    color: str


class Scene(_Model):
    id: str
    background: Background
    shapes: list[Shape] = Field(default_factory=list)
    item_layers: list[ItemLayer] = Field(default_factory=list)
    hotspots: list[Hotspot] = Field(default_factory=list)
    exits: list[Hotspot] = Field(default_factory=list)


class Goal(_Model):
    flag: str
    description: str | None = None


class MapsTo(_Model):
    item: str | None = None
    flag: str | None = None
    hotspot: str | None = None

    @model_validator(mode="after")
    def _exactly_one(self) -> MapsTo:
        chosen = [name for name in ("item", "flag", "hotspot") if getattr(self, name) is not None]
        if len(chosen) != 1:
            raise ValueError("maps_to must set exactly one of item, flag, hotspot")
        return self


class PuzzleNode(_Model):
    id: str
    kind: Literal["action", "goal"]
    maps_to: MapsTo | None = None


class PuzzleGraph(_Model):
    nodes: list[PuzzleNode]
    edges: list[tuple[str, str]]


class World(_Model):
    id: str
    title: str
    start_scene: str
    goal: Goal
    scenes: list[Scene]
    puzzle_graph: PuzzleGraph
