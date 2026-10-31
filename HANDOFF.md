# Handoff — Hotspot Phase 0

## Built

World schema (Pydantic, extra fields forbidden), JSON loader, hashable state, engine, BFS solver, publishability checks, Pillow renderer with four fidelity tiers, episode harness with JSONL logs, `random` and `oracle` agents, and two worlds: `tiny_key_door` (2 steps) and `rope_and_well` (7 steps). Tests in `tests/` are the contract.

## Decisions the brief did not spell out

- Exits are extra hotspots, hit-tested after the scene list. The last polygon that contains the point wins.
- One verb key per hotspot. The coin's optional use is its own hotspot (`wishing_niche`), not a second `use` on the well.
- `Requires` is a conjunction, and it has a single `has_item` and a single `flag`. The combine step is `use rope` on the workbench, which also requires `has_item: bucket` and `flag: holding_hook` (set when the hook is taken).
- Puzzle node ids map by exact item/flag/hotspot id, `get_<item>`, `make_<item>`, or token reversal onto a flag (`open_shed` ↔ `shed_open`). Edges are checked against sticky achievement on every reachable path, so consuming an item does not forget that you found it.
- Coverage "required" hotspots are the targets on the optimal path. `clicked` / `zoomed` / `used_correctly` separate never looked, looked, and used for a state change.
- Zoom does not change state. The engine records the box; the harness renders the crop from a higher-resolution paint. Invalid actions return the text `Nothing happens.`
- Tiers do not share one bitmap: 320×200, 640×480, and 1280×720 have different aspect ratios, so a 16:9 master would squash. Each tier is painted from the same normalized geometry at an integer scale and nearest-neighbour downscaled, then quantized with no dither. Images that already use fewer colours than the cap are left as-is, which keeps the key and the rope visible at `retro_320`.
- The example key was 22×12 px at `retro_320` and disappeared into the grass. It is now a 70×44 px gold block, and the shed has a darker door shape so the clickable door is visible. Hotspot ids and the 2-step path are unchanged. The coin niche is a black crack so the red herring is visible too.

## Known gaps

- No LLM agents, image-model art, authoring UI, or MCP server (out of scope).
- No generic inventory-to-inventory craft UI; combine is a hotspot `use`.
- Solver, dead-end search, and puzzle-graph search each stop at 100k states.
- Pixel observations hide hotspot names. The oracle is allowed to turn the symbolic path into centroids because it already knows the world.
- Debug overlay is a test/screenshot aid. Do not feed it to agents.

## Verify

```bash
python -m pip install -e ".[dev]"
pytest
python -m hotspot.harness --world worlds/rope_and_well.json --agent oracle --mode pixel
python -m hotspot.render
```

Do not weaken a test to make it pass. `validate()` returning no errors is the publish gate.
