"""Substrate pattern 1 — one INR per renderable weight.

The simplest canonical use of ``loom.render``: every renderable leaf in the
target pytree gets its own independent ondes ``SIREN`` body. ``params`` is
the dict of INRs keyed by ``KeyPath``; the renderer function looks up the
INR for the current path, evaluates it on a per-leaf coordinate grid, and
reshapes back to the leaf's expected shape.

FWS-side relevance: this is the **independent-INR baseline** the v3
programme studies before introducing shared bodies (pattern 2) or
hypernetworks (pattern 3). Each weight tensor's parameter count under loom
is the size of its dedicated INR rather than the size of the weight itself —
the FWS framing is that *if* the weight has spectral structure, the INR will
be much smaller than the weight. This script just verifies the substrate
plumbing: shapes, dtypes, gradient flow, no opinions about whether that
spectral compression actually happens for any given target.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from ondes import SIREN

import loom


class TinyMLP(eqx.Module):
    """Two-layer MLP used as the target pytree (~50 floats once tiny dims chosen)."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, *, key):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(in_features=4, out_features=6, key=k1)
        self.fc2 = eqx.nn.Linear(in_features=6, out_features=2, key=k2)

    def __call__(self, x):
        return self.fc2(jax.nn.relu(self.fc1(x)))


def make_coord_grid(shape: tuple[int, ...]) -> jax.Array:
    """Build a flat (prod(shape), rank) grid of normalised coords in ``[-1, 1]``.

    loom does not own coord-grid construction (Principle 5 — mechanism, not
    policy), so this helper lives in user code. The mesh is plain linspace
    per-axis; users with shape-1 axes get a single coord at 0.
    """
    axes = [jnp.linspace(-1.0, 1.0, n) if n > 1 else jnp.zeros(1) for n in shape]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([m.ravel() for m in mesh], axis=-1)


def main() -> None:
    key = jax.random.key(0)
    k_target, k_inrs = jax.random.split(key)

    target = TinyMLP(key=k_target)

    # Filter to floating-point arrays so we only build INRs for trainable weights
    # (the canonical pattern from PHILOSOPHY.md §"Composition recipes").
    def is_float(x):
        return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)

    renderable, passthrough = eqx.partition(target, is_float)

    # One INR per renderable leaf, keyed by a stable string tag derived from
    # the KeyPath. We use string tags rather than raw paths because JAX dicts
    # must have orderable keys for `jax.grad` to sort the leaves; KeyPath
    # parts (e.g. `GetAttrKey`) are not orderable. This is the
    # refactor-safe-dispatch recipe from PHILOSOPHY §"Refactor-safe dispatch"
    # — a five-line user pattern, not a loom primitive.
    def tag_of(path):
        return "/".join(getattr(p, "name", str(p)) for p in path)

    leaves_with_paths = jax.tree_util.tree_leaves_with_path(renderable)
    keys = jax.random.split(k_inrs, len(leaves_with_paths))
    inrs = {
        tag_of(path): SIREN(in_dim=len(leaf.shape), hidden_dim=8, num_hidden_layers=2, key=k)
        for (path, leaf), k in zip(leaves_with_paths, keys, strict=True)
    }

    def f(path, shape, dtype, params):
        coords = make_coord_grid(shape)
        flat = jax.vmap(params[tag_of(path)])(coords).astype(dtype)
        return flat.reshape(shape)

    rendered_renderable = loom.render(renderable, f, inrs)
    rendered = eqx.combine(rendered_renderable, passthrough)

    # Substrate invariants: shape preserved leaf-by-leaf, forward pass runs.
    assert rendered.fc1.weight.shape == target.fc1.weight.shape
    assert rendered.fc2.weight.shape == target.fc2.weight.shape

    x = jnp.ones(4)
    y_target = target(x)
    y_rendered = rendered(x)
    print(f"target output: {y_target}")
    print(f"rendered output: {y_rendered}")
    print(f"forward-pass output shape match: {y_target.shape == y_rendered.shape}")

    # Confirm gradients flow back into the INR params (Guarantee 5).
    def loss_of(p):
        r = eqx.combine(loom.render(renderable, f, p), passthrough)
        return jnp.sum(r(x) ** 2)

    grads = jax.grad(loss_of)(inrs)
    for tag, g in grads.items():
        norm = jnp.linalg.norm(g.layers[0].W)
        print(f"grad norm on {tag} (first SIREN layer W): {norm:.4f}")


if __name__ == "__main__":
    main()
