"""Run one episode per (world, seed) and write results.csv.

Agents: ``oracle``, ``random``, ``llm:<provider>:<model>`` with provider
``anthropic``, ``openai``, or ``xai``. ``--max-cost-usd`` stops before the
next episode would cross the budget. Oracle and random cost nothing.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from hotspot.harness import COVERAGE_CLASSES, run_episode
from hotspot.llm.pricing import estimate_cost
from hotspot.loader import load_world
from hotspot.render import TIERS

COLUMNS = [
    "world",
    "template",
    "mode",
    "tier",
    "dials",
    "model",
    "seed",
    "completed",
    "actions_used",
    "optimal_length",
    "efficiency",
    "wasted_actions",
    "zoom_count",
    "graph_progress",
    "never_looked",
    "looked_but_missed",
    "found_but_misused",
    "used_correctly",
    "tokens",
    "est_cost_usd",
]


def run_batch(
    worlds_dir: str | Path,
    agent: str,
    mode: str,
    tier: str,
    seeds: int,
    out_dir: str | Path,
    max_cost_usd: float | None = None,
) -> dict:
    if seeds < 1:
        raise ValueError("seeds must be at least 1")
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    catalog = _catalog(worlds_dir)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    log_dir = destination / "logs"
    rows = []
    spent = 0.0
    stopped = False
    floor = _floor_cost(agent)
    for world, meta in catalog:
        for seed in range(seeds):
            if max_cost_usd is not None and spent + floor > max_cost_usd:
                stopped = True
                break
            player = _make_agent(agent, world, seed, mode)
            metrics = run_episode(world, player, mode=mode, tier=tier, log_dir=log_dir)
            cost = metrics.get("est_cost_usd")
            spent += float(cost) if cost else (floor if agent.startswith("llm:") else 0.0)
            rows.append(_row(metrics, meta, seed))
        if stopped:
            break
    csv_path = destination / "results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    summary = _summary(rows, stopped, csv_path, spent)
    print(_format_summary(summary))
    return summary


def _catalog(worlds_dir: str | Path) -> list[tuple]:
    root = Path(worlds_dir)
    manifest = {}
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in payload.get("worlds", []):
            manifest[row["id"]] = row
    catalog = []
    for path in sorted(root.glob("*.json")):
        if path.name == "manifest.json":
            continue
        world = load_world(path)
        meta = manifest.get(world.id, {})
        catalog.append((world, meta))
    if not catalog:
        raise FileNotFoundError(f"no worlds in {root}")
    return catalog


def _make_agent(spec: str, world, seed: int, mode: str):
    if spec == "oracle":
        from hotspot.agents.oracle_agent import OracleAgent

        return OracleAgent(world)
    if spec == "random":
        from hotspot.agents.random_agent import RandomAgent

        return RandomAgent(seed=seed)
    if spec.startswith("llm:"):
        parts = spec.split(":")
        if len(parts) < 3:
            raise ValueError("llm agent must look like llm:<provider>:<model>")
        provider, model = parts[1], ":".join(parts[2:])
        from hotspot.agents.llm_agent import LLMAgent

        return LLMAgent(_client(provider, model), mode=mode, world=world, seed=seed)
    raise ValueError(f"unknown agent {spec!r}")


def _client(provider: str, model: str):
    if provider == "anthropic":
        from hotspot.llm.anthropic import AnthropicClient

        return AnthropicClient(model)
    if provider == "openai":
        from hotspot.llm.openai_compat import OpenAICompatibleClient

        return OpenAICompatibleClient(model)
    if provider == "xai":
        from hotspot.llm.openai_compat import xai_client

        return xai_client(model)
    raise ValueError(f"unknown llm provider {provider!r}")


def _floor_cost(spec: str) -> float:
    if spec in ("oracle", "random"):
        return 0.0
    if not spec.startswith("llm:"):
        return 0.0
    model = ":".join(spec.split(":")[2:])
    estimate = estimate_cost(model, 1500, 300)
    if estimate is None:
        return 0.01
    return max(estimate, 1e-6)


def _row(metrics: dict, meta: dict, seed: int) -> dict:
    summary = metrics.get("coverage_summary") or {}
    cost = metrics.get("est_cost_usd")
    dials = meta.get("dials", None)
    return {
        "world": metrics["world"],
        "template": meta.get("template", ""),
        "mode": metrics["mode"],
        "tier": metrics["tier"],
        "dials": "null" if dials is None else json.dumps(dials, sort_keys=True),
        "model": metrics.get("model") or "",
        "seed": seed,
        "completed": metrics["completed"],
        "actions_used": metrics["actions_used"],
        "optimal_length": metrics["optimal_length"],
        "efficiency": metrics["efficiency"],
        "wasted_actions": metrics["wasted_actions"],
        "zoom_count": metrics["zoom_count"],
        "graph_progress": metrics["graph_progress"],
        "never_looked": summary.get("never_looked", 0),
        "looked_but_missed": summary.get("looked_but_missed", 0),
        "found_but_misused": summary.get("found_but_misused", 0),
        "used_correctly": summary.get("used_correctly", 0),
        "tokens": int(metrics.get("total_input_tokens", 0)) + int(metrics.get("total_output_tokens", 0)),
        "est_cost_usd": "" if cost is None else cost,
    }


def _summary(rows: list[dict], stopped: bool, csv_path: Path, spent: float) -> dict:
    total = len(rows)
    completed = sum(1 for row in rows if str(row["completed"]) in ("True", "true", "1") or row["completed"] is True)
    efficiency = sum(float(row["efficiency"]) for row in rows) / total if total else 0.0
    coverage = {name: sum(int(row[name]) for row in rows) for name in COVERAGE_CLASSES}
    return {
        "rows": total,
        "completion_rate": (completed / total) if total else 0.0,
        "mean_efficiency": efficiency,
        "coverage": coverage,
        "stopped": stopped,
        "spent_usd": spent,
        "csv": str(csv_path),
    }


def _format_summary(summary: dict) -> str:
    coverage = " ".join(f"{name}={count}" for name, count in summary["coverage"].items())
    lines = [
        f"episodes: {summary['rows']}",
        f"completion: {summary['completion_rate']:.1%}",
        f"mean efficiency: {summary['mean_efficiency']:.3f}",
        f"coverage: {coverage}",
        f"spent_usd: {summary['spent_usd']:.6f}",
        f"csv: {summary['csv']}",
    ]
    if summary["stopped"]:
        lines.append("stopped: max-cost-usd")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Batch-run Hotspot episodes")
    parser.add_argument("--worlds", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--mode", choices=("symbolic", "pixel"), default="pixel")
    parser.add_argument("--tier", default="retro_320")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-cost-usd", type=float, default=None)
    args = parser.parse_args(argv)
    out = args.out or f"runs/batch_{time.strftime('%Y%m%dT%H%M%S')}"
    summary = run_batch(
        args.worlds,
        args.agent,
        args.mode,
        args.tier,
        args.seeds,
        out,
        max_cost_usd=args.max_cost_usd,
    )
    return 0 if summary["rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
