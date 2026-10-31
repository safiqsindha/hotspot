"""Load a world JSON file and reject dangling ids."""

from __future__ import annotations

import json
from pathlib import Path

from hotspot.schema import World
from hotspot.validate import Issue, reference_issues


class DanglingReferenceError(Exception):
    def __init__(self, issues: list[Issue]):
        self.issues = issues
        detail = "; ".join(f"{issue.code}: {issue.message}" for issue in issues)
        super().__init__(detail)


def parse_world(data: dict) -> World:
    world = World.model_validate(data)
    issues = reference_issues(world)
    if issues:
        raise DanglingReferenceError(issues)
    return world


def load_world(path: str | Path) -> World:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_world(raw)
