# Examples

Six canonical patterns the loom substrate must support without forking. Each
example is a self-contained script that exercises one row of
[§"Test the substrate"](../docs/PHILOSOPHY.md#test-the-substrate) in the
philosophy doc, and prints `PASS` on success.

## What these are (and what they are not)

These examples are **API-composition demonstrations and smoke tests**, not
baseline-comparison experiments. They show that `loom.render(P, f, params)`
composes cleanly with the relevant JAX primitives (`jax.vmap`, `jax.lax.scan`,
`jax.grad`, `eqx.partition`) without the substrate needing to fork for any of
the six patterns. Each script also includes a *within-loom* contrast smoke
test (e.g. distinct-vs-replicated INRs) confirming the pattern produces
differentiated output where it claims to.

They are **not** evidence that loom-rendered training is competitive with
direct training of the target network. That claim requires K-seed training on
real tasks with paired statistical analysis, and lives in the fws repo once
that programme runs — not here.

The examples print contrasted numerical outputs (config A versus config B)
for the reader to observe. They do **not** assert against specific magnitudes
— initial conditions and hyperparameters dominate single-shot numbers. Only
structural invariants are asserted (e.g. shape correctness, pure-`f`
determinism producing identically-zero deviation under replicated inputs).

For the formal substrate guarantees (the seven contract invariants), see
[`tests/test_contract.py`](../tests/test_contract.py).

## Index

| # | Pattern | Script | Philosophy case |
|---|---|---|---|
| 1 | One INR per weight | [`01_one_inr_per_weight.py`](01_one_inr_per_weight.py) | [case 1](../docs/PHILOSOPHY.md#test-the-substrate) |
| 2 | Shared functa body with per-leaf FiLM | [`02_shared_functa_film.py`](02_shared_functa_film.py) | [case 2](../docs/PHILOSOPHY.md#test-the-substrate) |
| 3 | Hypernet over a target distribution (vmap) | [`03_hypernet_distribution.py`](03_hypernet_distribution.py) | [case 3](../docs/PHILOSOPHY.md#test-the-substrate) |
| 4 | Hypernet with per-target context (scan) | [`04_hypernet_conditioning.py`](04_hypernet_conditioning.py) | [case 4](../docs/PHILOSOPHY.md#test-the-substrate) |
| 5 | Inner-loop adaptation (iMAML / SAM / Reptile) | [`05_inner_loop_adaptation.py`](05_inner_loop_adaptation.py) | [case 5](../docs/PHILOSOPHY.md#test-the-substrate) |
| 6 | Heterogeneous `f` per leaf-group | [`06_heterogeneous_f.py`](06_heterogeneous_f.py) | [case 6](../docs/PHILOSOPHY.md#test-the-substrate) |

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
