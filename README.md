# Hotspot

Hotspot builds small point-and-click adventure worlds, proves they are solvable, and lets agents play them. A world is a set of static scenes: a background, flat shapes, and item layers that show or hide. Clickable regions (hotspots) sit on top. Puzzles are inventory chains, and the designer's intended structure is a puzzle dependency graph. Geometry is normalized, so every pixel and hotspot boundary is known exactly.

Agents play in two modes:

- **symbolic** — the agent sees hotspot names and chooses a verb
- **pixel** — the agent sees a rendered image and clicks normalized coordinates

## Install

Python 3.11 or newer. The core package needs only pydantic and pillow.

```bash
python -m pip install -e ".[dev]"
python -m pip install -e ".[llm]"   # same code; clients use the stdlib
```

## Run the tests

```bash
pytest
```

The live model smoke test stays skipped unless `HOTSPOT_LIVE_LLM=1`.

## Run an episode

```bash
python -m hotspot.harness --world worlds/rope_and_well.json --agent oracle --mode pixel
python -m hotspot.harness --world worlds/tiny_key_door.json --agent random --mode symbolic --seed 1
```

The harness prints episode metrics. Optimal length comes from the solver. The default action cap is `max(20, 3 × optimal length)`. Each step is appended to a JSONL file under `runs/` (gitignored).

Zoom has its own budget (`zoom_budget`, default 15). A zoom inside the budget does not increment `actions_used` or `wasted_actions`. The next zoom returns "Nothing happens." and is logged. Metrics and every JSONL line include `zoom_count` and `zoom_budget`.

Required hotspots (targets on the optimal path) end in one of four classes: `never_looked`, `looked_but_missed`, `found_but_misused`, `used_correctly`. The old `clicked`, `zoomed`, and `used_correctly` booleans are still on each hotspot. `coverage_summary` counts the four classes.

Regenerate the tier screenshots with `python -m hotspot.render`. Images land in `docs/screens/`.

## Generate worlds

```bash
python -m hotspot.generate --count 8 --seed 1 --out worlds/generated
python -m hotspot.contact
```

Templates: `key_door`, `tool_obstacle`, `two_scene_fetch`, `combine`, `chain3`, `red_herring`. Names and layouts come from the seed. A candidate is kept only when `validate()` reports no errors, optimal length is 2–12, and its signature (template + item names + layout) is new. The command fails if it cannot fill `--count`, and says how many were accepted.

`worlds/generated/manifest.json` lists id, template, seed, optimal length, scene count, and dials (`null` unless you dial the world yourself).

A red herring is an item layer with `"decoy": true`. It can be taken and is exempt from the unused-item check. Every other given item must still be used.

Contact sheets (plain and debug, `retro_320` and `vga_640`) are written under `docs/generated/`.

## Apply dials

```python
from hotspot.dials import PRESETS, apply_dials
from hotspot.loader import load_world

world = load_world("worlds/rope_and_well.json")
harder = apply_dials(world, seed=1, **PRESETS["medium"])
```

`apply_dials` returns a new world and does not mutate the input. Same seed and arguments are byte-identical.

- `target_scale` (0.3–1.5) scales each item-layer shape and the hotspot that requires that layer, about the centroid. If the result fails `validate()`, `DialError` is raised with the issues. Nothing is silently grown back.
- `decoy_shapes` adds flat shapes with no hotspot, drawn under item layers, never overlapping a hotspot in that scene.
- `decoy_hotspots` adds look-only "Just a rock." regions that give nothing and set nothing.
- `color_similarity` (0–1) pulls decoy colors toward the required item colors.

Presets `easy`, `medium`, and `hard` are in `hotspot.dials.PRESETS`. `python -m hotspot.dials` renders `rope_and_well` at `retro_320` into `docs/dials/`. Fidelity tier is a render argument, not a dial.

## Run a batch

```bash
python -m hotspot.batch \
  --worlds worlds/generated \
  --agent oracle \
  --mode pixel \
  --tier retro_320 \
  --seeds 1 \
  --out runs/batch_oracle
```

Agents: `oracle`, `random`, and `llm:<provider>:<model>` (`anthropic`, `openai`, `xai`). One episode per world per seed (`0 .. seeds-1`). `results.csv` has world, template, mode, tier, dials, model, seed, completed, actions_used, optimal_length, efficiency, wasted_actions, zoom_count, graph_progress, the four coverage-class counts, tokens, and est_cost_usd. The process prints completion rate, mean efficiency, and coverage totals.

`--max-cost-usd` refuses to start an episode that would cross the budget. Oracle and random cost nothing. Unknown model prices stay blank, not zero. Prices live in `hotspot/llm/prices.json` (USD per million tokens).

## Add a model client

```python
from hotspot.agents.llm_agent import LLMAgent
from hotspot.llm.scripted import ScriptedClient

agent = LLMAgent(ScriptedClient(['{"verb":"look","target":"key_spot"}']), mode="symbolic", world=world)
```

`ModelClient.complete(system, messages, images, max_tokens, temperature)` returns a `ModelReply` (text, token counts, latency, raw). Ship adapters:

- `hotspot.llm.anthropic.AnthropicClient` — `ANTHROPIC_API_KEY`
- `hotspot.llm.openai_compat.OpenAICompatibleClient` — `OPENAI_API_KEY`, optional `base_url`
- `hotspot.llm.openai_compat.xai_client` — `XAI_API_KEY`, `https://api.x.ai/v1`

Keys are read from the environment only and are not logged or included in `repr`. One model call per step. The reply is one JSON action; fences and leading prose are stripped. Up to two silent re-asks are logged (`parse_retries`) and do not count as actions. After that the step is an invalid look. Symbolic prompts list visible hotspot names and verbs. Pixel prompts send the image and never list hotspot names. Neither prompt includes the optimal path, the puzzle graph, polygons, or items that are not already in the inventory. The last 8 actions and result texts are included. Default temperature is 0. Model, temperature, and seed are on every JSONL line.

## Add a world

1. Add a JSON file under `worlds/`. Coordinates are normalized: `x` right, `y` down, both in `0–1`.
2. Load it with `hotspot.loader.load_world`. Unknown fields are rejected. Ids referenced by effects and requirements must exist.
3. Run `hotspot.validate.validate(world)`. A world is publishable only when that returns no errors: the goal is reachable, no reachable state is a dead end, every given item is used (unless its layer is `decoy`), every hotspot can be clicked, hotspot polygons have area at least `0.002` and lie inside the scene, and the puzzle graph is an acyclic map onto real actions.
4. `hotspot.solver.solve(world)` returns the optimal symbolic path.

Shipped worlds:

| World | Optimal length | What it checks |
| --- | --- | --- |
| `worlds/tiny_key_door.json` | 2 | Take a key, use it on a door |
| `worlds/rope_and_well.json` | 7 | Two scenes, three useful items, one combine, one red-herring coin |

### Actions

Symbolic: `{"verb": "use", "target": "shed_door", "item": "key"}`.

Pixel: `{"verb": "use", "x": 0.72, "y": 0.6, "item": "key"}`. The point hits the topmost active hotspot. Exits are hotspots listed after the scene's hotspots, so they win ties.

Zoom (pixel, no state change): `{"verb": "zoom", "box": [x0, y0, x1, y1]}`.

Verbs are `look`, `take`, `use`, `talk`, and `go`. `use` may require `with_item`, which must already be in the inventory. Requirements (`flag`, `not_flag`, `has_item`, `layer_visible`) are a conjunction. An invalid action returns "Nothing happens." and leaves the state unchanged.

### Puzzle graph

Prefer `maps_to` on every node, exactly one of `{"item": id}`, `{"flag": id}`, or `{"hotspot": id}`. A dangling id is an error. A hotspot mapping counts only after a state-changing use of that hotspot. If `maps_to` is omitted, the loader still accepts the world and `validate` emits a warning (`legacy_node_mapping`) and falls back to the old names: exact id, `get_<item>`, `make_<item>`, or a token-reversed flag (`open_shed` ↔ `shed_open`).

`goal.description` is the sentence shown to an agent. If it is missing, the world title is used.

Edges mean "the parent must be achieved before the child." Achievement sticks for the rest of the path, so spending the key still counts as having got it.

## Add an agent

An agent is an object with `act(observation) -> action`.

Symbolic observations carry `scene`, `text`, `inventory`, and `hotspots` (`id`, `name`, `verbs`). Pixel observations carry `image` (a PIL image), `inventory`, and `text`, and do not list hotspot names. `observation.mode` is `"symbolic"` or `"pixel"`.

```python
class LookAgent:
    def act(self, observation):
        if observation.mode == "symbolic" and observation.hotspots:
            return {"verb": "look", "target": observation.hotspots[0]["id"]}
        return {"verb": "look", "x": 0.5, "y": 0.5}
```

Pass it to `hotspot.harness.run_episode(world, agent, mode="symbolic")`. `OracleAgent` replays the solver. `RandomAgent` samples until the cap. `LLMAgent` is described above.

## Layout

```text
hotspot/           schema, engine, solver, validator, renderer, harness, dials, generate, batch
hotspot/agents/    random, oracle, llm
hotspot/llm/       Anthropic, OpenAI-compatible, scripted client, prices.json
worlds/            shipped and generated JSON
tests/             the contract
docs/screens/      tier screenshots
docs/dials/        rope_and_well easy / medium / hard at retro_320
docs/generated/    contact sheets
runs/              episode logs and batch output (gitignored)
```

Fidelity tiers: `retro_320` (320×200, at most 16 colours), `vga_640` (640×480, at most 256), `hd_1280` (1280×720), `uhd_3840` (3840×2160). The scene is painted from normalized geometry, nearest-neighbour downscaled, then quantized. Pass `debug=True` only for screenshots and tests — never to an agent.
