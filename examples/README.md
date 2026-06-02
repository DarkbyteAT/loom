# Examples

Six canonical patterns the loom substrate must support without forking. Each
example is a self-contained script that exercises one row of
[§"Test the substrate"](../docs/PHILOSOPHY.md#test-the-substrate) in the
philosophy doc, and prints `PASS` on success.

| # | Pattern | Script | Philosophy case |
|---|---|---|---|
| 1 | One INR per weight | [`01_one_inr_per_weight.py`](01_one_inr_per_weight.py) | [case 1](../docs/PHILOSOPHY.md#test-the-substrate) |
| 2 | Shared functa body with per-leaf FiLM | [`02_shared_functa_with_film.py`](02_shared_functa_with_film.py) | [case 2](../docs/PHILOSOPHY.md#test-the-substrate) |
| 3 | Hypernet over a target distribution (vmap) | [`03_hypernet_vmap.py`](03_hypernet_vmap.py) | [case 3](../docs/PHILOSOPHY.md#test-the-substrate) |
| 4 | Hypernet with per-target context (scan) | [`04_hypernet_scan_contexts.py`](04_hypernet_scan_contexts.py) | [case 4](../docs/PHILOSOPHY.md#test-the-substrate) |
| 5 | Inner-loop adaptation (iMAML / SAM / Reptile) | [`05_inner_loop_adaptation.py`](05_inner_loop_adaptation.py) | [case 5](../docs/PHILOSOPHY.md#test-the-substrate) |
| 6 | Heterogeneous `f` per leaf-group | [`06_heterogeneous_per_leaf.py`](06_heterogeneous_per_leaf.py) | [case 6](../docs/PHILOSOPHY.md#test-the-substrate) |

## Running

```bash
uv run python examples/01_one_inr_per_weight.py
```

Run them all via the integration test suite:

```bash
uv run pytest tests/test_examples.py
```

If any of these patterns required a special primitive added to loom, or forced
the user to subclass anything, the substrate would have failed — see the
guarantee at the end of [§"Test the substrate"](../docs/PHILOSOPHY.md#test-the-substrate).
