"""Per-model prices in USD per million tokens. Unknown models cost None, not zero."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def price_table() -> dict:
    path = Path(__file__).with_name("prices.json")
    return json.loads(path.read_text(encoding="utf-8"))


def estimate_cost(model: str | None, input_tokens: int, output_tokens: int) -> float | None:
    if not model:
        return None
    row = price_table().get(model)
    if not row:
        return None
    return (
        input_tokens * float(row["input_per_million"]) + output_tokens * float(row["output_per_million"])
    ) / 1_000_000
