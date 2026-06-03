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

    # Filter to floating-point arrays with at least one axis so we only build
    # INRs for trainable, coordinate-bearing weights (the canonical pattern
    # from PHILOSOPHY.md §"Composition recipes"). The `ndim > 0` clause
    # skips 0-D scalars (e.g. learnable temperatures) — coordinate-based
    # INRs need at least one axis to build a grid against.
    def is_float(x):
        return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) and x.ndim > 0

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
        tag_of(path): SIREN(in_dim=leaf.ndim, hidden_dim=8, num_hidden_layers=2, key=k)
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

    # Use `eqx.filter_grad` rather than `jax.grad` — Equinox's filtered grad
    # differentiates only the floating-array leaves and passes everything else
    # through, so the call stays robust against static fields inside the
    # `SIREN` bodies (e.g. `out_features: int | None`).
    grads = eqx.filter_grad(loss_of)(inrs)
    for tag, g in grads.items():
        norm = jnp.linalg.norm(g.layers[0].W)
        print(f"grad norm on {tag} (first SIREN layer W): {norm:.4f}")

    # Contrast smoke test — does path dispatch actually distinguish leaves?
    #
    # Build a second `inrs_replicated` dict where the same-rank `weight` tag
    # maps to the SAME `SIREN` instance as `fc1/weight` (rather than its own
    # independent network), then render once with each configuration. We
    # compare the two `weight` leaves (both rank-2) via cosine similarity on
    # the shared prefix of their flat representations — shape-agnostic and
    # bounded in [-1, 1], so the numbers are interpretable.
    #
    # No threshold is asserted: initial conditions and hyperparameters
    # dominate any single number. The structural claim is "the two
    # configurations produce different renderings", and the evidence for
    # that claim is the reader observing the two numbers side-by-side.
    print("\n--- contrast smoke test: path dispatch vs replicated INR ---")

    rank2_tags = sorted(t for t, leaf in inrs.items() if leaf.layers[0].W.shape[1] == 2)
    assert rank2_tags == ["fc1/weight", "fc2/weight"], rank2_tags

    shared_inr = inrs[rank2_tags[0]]
    inrs_replicated = {**inrs, rank2_tags[1]: shared_inr}

    rendered_distinct = loom.render(renderable, f, inrs)
    rendered_replicated = loom.render(renderable, f, inrs_replicated)

    def flat(leaf_pytree, tag: str):
        attr1, attr2 = tag.split("/")
        return getattr(getattr(leaf_pytree, attr1), attr2).ravel()

    def cos_sim(a: jax.Array, b: jax.Array) -> float:
        n = min(a.size, b.size)
        a, b = a[:n], b[:n]
        return float(jnp.dot(a, b) / (jnp.linalg.norm(a) * jnp.linalg.norm(b) + 1e-12))

    a_d = flat(rendered_distinct, rank2_tags[0])
    b_d = flat(rendered_distinct, rank2_tags[1])
    a_r = flat(rendered_replicated, rank2_tags[0])
    b_r = flat(rendered_replicated, rank2_tags[1])

    print(f"with path dispatch:  cos(fc1.weight, fc2.weight) = {cos_sim(a_d, b_d):+.4f}")
    print(f"with replicated INR: cos(fc1.weight, fc2.weight) = {cos_sim(a_r, b_r):+.4f}")


if __name__ == "__main__":
    main()
    print("PASS")
