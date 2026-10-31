# Hotspot

Hotspot builds small point-and-click adventure worlds, proves they are solvable, and lets agents play them. A world is a set of static scenes: a background, flat shapes, and item layers that show or hide. Clickable regions (hotspots) sit on top. Puzzles are inventory chains, and the designer's intended structure is a puzzle dependency graph. Phase 0 worlds are drawn in code, so every pixel and hotspot boundary is known exactly.

Agents play in two modes:

- **symbolic** — the agent sees hotspot names and chooses a verb
- **pixel** — the agent sees a rendered image and clicks normalized coordinates

## Install

Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

## Run the tests

From the repository root:

```bash
pytest
```

## Run an episode

```bash
python -m hotspot.harness --world worlds/rope_and_well.json --agent oracle --mode pixel
python -m hotspot.harness --world worlds/tiny_key_door.json --agent random --mode symbolic --seed 1
```

The harness prints episode metrics. Optimal length comes from the solver. The default action cap is `max(20, 3 × optimal length)`, which stops an agent from trying every item on every hotspot. Each step is appended to a JSONL file under `runs/`.

Regenerate the tier screenshots (plain and debug overlay) with:

```bash
python -m hotspot.render
```

Images land in `docs/screens/`.

## Add a world

1. Add a JSON file under `worlds/`. Coordinates are normalized: `x` right, `y` down, both in `0–1`.
2. Load it with `hotspot.loader.load_world`. Unknown fields are rejected. Ids referenced by effects and requirements must exist.
3. Run `hotspot.validate.validate(world)`. A world is publishable only when that returns no errors: the goal is reachable, no reachable state is a dead end, every given item is used, every hotspot can be clicked, hotspot polygons have area at least `0.002` and lie inside the scene, and the puzzle graph is an acyclic map onto real actions.
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

Verbs are `look`, `take`, `use`, `talk`, and `go`. `use` may require `with_item`. Requirements (`flag`, `not_flag`, `has_item`, `layer_visible`) are a conjunction. An invalid action returns "Nothing happens." and leaves the state unchanged.

### Puzzle-graph ids

A node id maps onto the world when it matches any of:

- an item id, a flag id, or a hotspot id
- `get_<item>` or `make_<item>` for an item some effect gives
- the token-reversed form of a flag (`open_shed` ↔ `shed_open`, `raised_water` ↔ `water_raised`)

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

Pass it to `hotspot.harness.run_episode(world, agent, mode="symbolic")`. `OracleAgent` replays the solver. `RandomAgent` samples legal-looking clicks until the cap. Do not import an LLM client into this package in Phase 0; add a new module under `hotspot/agents/` when you do.

## Layout

```text
hotspot/          schema, engine, solver, validator, renderer, harness
hotspot/agents/   random_agent, oracle_agent
worlds/           JSON worlds
tests/            the contract
docs/screens/     tier screenshots, with and without the debug overlay
runs/             episode JSONL (gitignored)
```

Fidelity tiers, one world and several looks: `retro_320` (320×200, at most 16 colours), `vga_640` (640×480, at most 256), `hd_1280` (1280×720), `uhd_3840` (3840×2160). The scene is painted from normalized geometry, nearest-neighbour downscaled, then quantized. Pass `debug=True` only for screenshots and tests — never to an agent.
