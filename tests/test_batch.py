"""Batch runner writes one CSV row per (world, seed)."""

import csv
from pathlib import Path

from hotspot.batch import COLUMNS, run_batch
from hotspot.generate import generate_worlds, write_worlds


def test_oracle_and_random_batches_write_csv(tmp_path):
    worlds = tmp_path / "worlds"
    generated = generate_worlds(3, seed=2)
    write_worlds(generated, worlds, batch_seed=2)

    oracle = run_batch(worlds, "oracle", "pixel", "retro_320", seeds=1, out_dir=tmp_path / "oracle")
    assert oracle["rows"] == 3
    assert oracle["completion_rate"] == 1.0
    assert oracle["mean_efficiency"] == 1.0
    _assert_csv(Path(oracle["csv"]), 3)

    random_run = run_batch(worlds, "random", "symbolic", "retro_320", seeds=1, out_dir=tmp_path / "random")
    assert random_run["rows"] == 3
    _assert_csv(Path(random_run["csv"]), 3)


def _assert_csv(path: Path, rows: int) -> None:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == COLUMNS
        body = list(reader)
    assert len(body) == rows
    assert all(row["world"] and row["template"] for row in body)
    assert all(row["dials"] == "null" for row in body)
