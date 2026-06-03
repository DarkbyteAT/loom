# loom

A substrate for re-parameterising JAX pytrees by the output of arbitrary functions, in a way that is scannable and optimisable.

Not a renderer for neural-network weights. Not a hypernet framework. Not a training studio. A substrate. INRs from [`ondes`](https://github.com/DarkbyteAT/ondes) are the canonical instantiation, but loom does not bake them in — anything that maps `(path, shape, dtype, params) → array-of-leaf-shape` is a valid renderer.

See [`docs/PHILOSOPHY.md`](docs/PHILOSOPHY.md) for the full design rationale and the seven contract guarantees.

## What it looks like

```python
import loom

def f(path, shape, dtype, params):
    # Any pure function of (path, shape, dtype, params) returning a jax.Array
    # of exactly `shape` and `dtype`. INRs, hypernets, lookups — anything.
    ...

rendered = loom.render(P, f, params)
```

## Install

loom is not on PyPI yet. Install from git:

```bash
uv pip install git+https://github.com/DarkbyteAT/loom
```

## Quick start

```bash
git clone https://github.com/DarkbyteAT/loom
cd loom
uv pip install -e ".[examples]"
uv run python examples/01_one_inr_per_weight.py
```

The [`examples/`](examples/) directory contains six self-contained scripts, one per canonical substrate pattern (per-leaf INR, shared functa + FiLM, hypernet vmap/scan, inner-loop adaptation, heterogeneous-per-leaf). Each prints `PASS` on success. They are API-composition demonstrations and smoke tests, not benchmarks. See [`examples/README.md`](examples/README.md) for the full index.

## License

MIT
