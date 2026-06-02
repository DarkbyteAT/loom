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

See [`examples/README.md`](examples/README.md) for the full index, and
[`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the design rationale.

## License

MIT
