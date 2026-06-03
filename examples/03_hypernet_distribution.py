"""Substrate pattern 3 — one render call, B distinct param sets, via vmap.

``jax.vmap(loom.render, in_axes=(None, None, 0))`` over a leading batch axis
of ``params``. The target pytree ``P`` and the renderer function ``f`` are
broadcast (``None``); ``params`` is mapped (``0``). One call yields B
independent materialisations of the same target shape — each with its own
INR parameters.

FWS-side relevance: this is the **amortised hypernet pattern**. Instead of
training one INR per target (pattern 1) or one shared body across leaves of
a single target (pattern 2), here a *hypernetwork* emits the INR's params
from external conditioning. During training the user samples B contexts,
emits B param sets, materialises B distinct target weight pytrees with one
vmap'd render call, evaluates the downstream losses in parallel, and
backpropagates through every batch element at once. The substrate's
Guarantee 4 (vmap broadcast pattern) is what makes the "B different weight
sets" composition free.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from ondes import SIREN

import loom


BATCH = 4
HIDDEN_DIM = 8
NUM_HIDDEN_LAYERS = 2


class TinyMLP(eqx.Module):
    """Target pytree — same shape as examples 01/02."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, *, key):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(in_features=4, out_features=6, key=k1)
        self.fc2 = eqx.nn.Linear(in_features=6, out_features=2, key=k2)


def make_coord_grid(shape: tuple[int, ...]) -> jax.Array:
    """User-side coord grid (loom does not own this; Principle 5)."""
    axes = [jnp.linspace(-1.0, 1.0, n) if n > 1 else jnp.zeros(1) for n in shape]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([m.ravel() for m in mesh], axis=-1)


def main() -> None:
    key = jax.random.key(0)
    k_target, k_bodies = jax.random.split(key)

    target = TinyMLP(key=k_target)

    # `ndim > 0` skips 0-D scalars (e.g. learnable temperatures) — coordinate-
    # based INRs need at least one axis to build a grid against.
    def is_float(x):
        return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) and x.ndim > 0

    renderable, _ = eqx.partition(target, is_float)

    # Fix in_dim to the max rank across renderable leaves so a single shared
    # body can serve every leaf shape — same convention as example 02.
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(renderable)
    in_dim = max(len(leaf.shape) for _, leaf in leaves_with_paths)

    # Build B distinct SIREN bodies by splitting `BATCH` keys. In a real
    # hypernet the bodies would be *emitted* by a network from per-batch
    # conditioning; here we draw them directly from PRNG keys to keep the
    # example focused on the substrate's vmap behaviour.
    batch_keys = jax.random.split(k_bodies, BATCH)

    def make_body(k):
        return SIREN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, key=k)

    bodies_batch = jax.vmap(make_body)(batch_keys)

    def pad_coords(coords: jax.Array, target_in_dim: int) -> jax.Array:
        if coords.shape[-1] == target_in_dim:
            return coords
        pad = jnp.zeros((coords.shape[0], target_in_dim - coords.shape[-1]))
        return jnp.concatenate([coords, pad], axis=-1)

    def f(path, shape, dtype, params):
        # `params` here is *one* body — vmap strips the leading axis before
        # handing it to f. The renderer never sees the batch dim.
        coords = pad_coords(make_coord_grid(shape), in_dim)
        flat = jax.vmap(params)(coords).astype(dtype)
        return flat.reshape(shape)

    # The canonical batched-render call from PHILOSOPHY Guarantee 4:
    # P and f are broadcast (None); params is mapped (0).
    rendered_batch = jax.vmap(loom.render, in_axes=(None, None, 0))(renderable, f, bodies_batch)

    # One render call → B distinct CNN-shape parameter sets stacked on axis 0.
    print(f"fc1.weight stacked shape: {rendered_batch.fc1.weight.shape}")  # (BATCH, 6, 4)
    print(f"fc2.weight stacked shape: {rendered_batch.fc2.weight.shape}")  # (BATCH, 2, 6)
    assert rendered_batch.fc1.weight.shape == (BATCH, *target.fc1.weight.shape)
    assert rendered_batch.fc2.weight.shape == (BATCH, *target.fc2.weight.shape)

    # Sanity-check: the batch elements are genuinely *different* materialisations.
    # If they were identical, vmap would have collapsed and the hypernet would
    # be doing no work — print pairwise differences to show they actually differ.
    w = rendered_batch.fc1.weight
    pair_diffs = jnp.linalg.norm(w[0] - w[1:], axis=(-2, -1))
    print(f"‖fc1.weight[0] − fc1.weight[i]‖ for i in 1..{BATCH - 1}: {pair_diffs}")

    # Contrast smoke test — does vmap-over-params actually distribute, or is
    # it covertly broadcasting?
    #
    # Build a second batch where every batch element is the SAME body (the
    # first one replicated B times) and render through the same vmap call.
    # Under correct vmap semantics the per-batch pair-diffs must collapse to
    # zero up to floating-point rounding — this is a structural invariant of
    # `jax.vmap`, not a property of initial conditions, so we assert it
    # (with a dtype-derived tolerance — see the assert site below). The
    # distinct-params line alongside is reported for the reader to observe;
    # we make no claim about its magnitude.
    print("\n--- contrast smoke test: distinct params vs replicated params ---")

    # `eqx.partition` splits arrays from anything else (static fields, `None`
    # leaves, plain Python scalars) so the broadcast only touches the array
    # branch. Without the split, a future non-array dynamic leaf in `ondes`
    # would TypeError under the slice — this is the standard Equinox pattern
    # for "do something array-only to a pytree of mixed leaves."
    arrays_batch, static = eqx.partition(bodies_batch, eqx.is_array)
    arrays_replicated = jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[0:1], (BATCH, *x.shape[1:])),
        arrays_batch,
    )
    bodies_replicated = eqx.combine(arrays_replicated, static)

    rendered_batch_replicated = jax.vmap(loom.render, in_axes=(None, None, 0))(renderable, f, bodies_replicated)
    w_r = rendered_batch_replicated.fc1.weight
    pair_diffs_replicated = jnp.linalg.norm(w_r[0] - w_r[1:], axis=(-2, -1))

    print(f"with distinct params:   pair-diffs = {pair_diffs}")
    print(f"with replicated params: pair-diffs = {pair_diffs_replicated}")

    # Tolerance is derived from leaf dtype, not chosen by intuition, so the
    # assertion transfers from float32 to float16 / float64 without silent
    # breakage. N=64 leaves headroom for the handful of fp ops between
    # `loom.render`'s input and the norm-of-difference here (SIREN forward
    # pass, broadcast subtraction, two-axis reduction).
    eps = jnp.finfo(pair_diffs_replicated.dtype).eps
    assert jnp.all(pair_diffs_replicated < 64 * eps), (
        "vmap is fabricating diversity not present in params — replicated-slice "
        "outputs should be identical up to floating-point rounding"
    )


if __name__ == "__main__":
    main()
    print("PASS")
