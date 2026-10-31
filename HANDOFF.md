# Handoff — Hotspot Phase 1

## Built

Phase 0 (schema, solver, renderer, oracle/random harness, two worlds) plus: one `requirements_met` used by `step` and the solver; four end-of-episode coverage classes; a separate zoom budget; explicit `maps_to` on puzzle nodes; an LLM agent (Anthropic, OpenAI-compatible including xAI, and `ScriptedClient`); difficulty dials; a six-template generator; contact sheets; a batch CSV runner. Core deps are still pydantic and pillow. `pip install -e ".[llm]"` adds no packages.

## Decisions the brief did not spell out

- `requirements_met(spec, state, action=None)` lives in `engine.py`. `with_item` must already be in the inventory, and the action's `item` must match. The solver only emits actions that change state.
- Coverage class priority is `used_correctly` > `found_but_misused` > `looked_but_missed` > `never_looked`. `used_correctly` means every optimal-path `(verb, item)` for that hotspot was issued (symbolic target or pixel hit). A state change is not required. Old booleans remain.
- Zooms inside the budget do not touch `actions_used`, `wasted_actions`, or the action cap. The 16th zoom (budget 15) is "Nothing happens." After another budget's worth of free refusals, further refusals count as wasted so a zoom-only agent still stops.
- `maps_to` is exactly one of item, flag, hotspot. Absent `maps_to` warns `legacy_node_mapping` and uses the Phase 0 names. A hotspot mapping is true only after a state-changing hit. `goal.description` falls back to the title.
- Malformed model output gets two silent re-asks, then `{"verb":"look"}` (a wasted step). History is the last 8 actions. Unknown models price as `null`, not 0. Keys never enter logs or `repr`.
- Dials JSON-copy the world. Only item-layer shapes and hotspots with `requires_layer_visible` scale. `DialError` carries the validate issues. Decoys stay in-scene and off every hotspot. Presets: easy 1.1/1/0, medium 0.9/3/2/0.35, hard 0.6/5/3/0.85. Medium `rope_and_well` at `retro_320` still shows the rope and the coin.
- `item_layers[].decoy: true` exempts that item from `unused_item`. `red_herring` uses it. Generator slots are five disjoint rectangles, mirrored three ways, so required sprites stay about 90×70 px at `retro_320`. Signature is template + item names + layout. Manifest `dials` is null.
- Batch `--max-cost-usd` uses a floor of 0 for oracle/random and one priced call (or $0.01 if unpriced) for an LLM, and will not start an episode that would cross it.

## Known gaps

- No live model call was made. Set `HOTSPOT_LIVE_LLM=1` for the skipped smoke test. No image-model art, MCP, or web UI.
- Generator output is not dialled. Solver and graph search still stop at 100k states.
- There is no inventory-to-inventory craft UI; combine is a hotspot `use`.

## Verify

```bash
python -m pip install -e ".[dev]"
pytest
python -m hotspot.generate --count 8 --seed 1 --out worlds/generated
python -m hotspot.batch --worlds worlds/generated --agent oracle --mode pixel --tier retro_320 --seeds 1
python -m hotspot.dials
python -m hotspot.contact
```

Oracle pixel on the eight generated worlds is 100% completion at efficiency 1.0. Do not weaken a Phase 0 test. `validate()` with no errors is the publish gate.
