# loom

Implicit neural renderer for parameterising neural network weights as low-dimensional manifolds

## Installation

```bash
pip install loom
```

## Quick start

The [`examples/`](examples/) directory contains six self-contained scripts, one
per canonical substrate pattern (per-leaf INR, shared functa + FiLM, hypernet
vmap/scan, inner-loop adaptation, heterogeneous-per-leaf). Each prints `PASS`
on success.

```bash
uv run python examples/01_one_inr_per_weight.py
```

These are API-composition demonstrations and smoke tests, not benchmarks —
they show that the substrate composes with JAX primitives, not that
loom-rendered training is competitive with direct training. See
[`examples/README.md`](examples/README.md) for the scope statement and full
index, and [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the design rationale.

## License

MIT
