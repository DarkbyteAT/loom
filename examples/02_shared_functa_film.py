"""Substrate pattern 2 — shared functa body with per-leaf FiLM modulation.

One ondes ``SIREN`` body shared across every renderable leaf; per-leaf
identity is carried by a small FiLM tensor (gamma, beta per hidden layer).
loom dispatches on path: the renderer looks up the FiLM for the current
leaf and conditions the shared body on it. Most parameters live in the
*body* (shared); only the FiLM is per-leaf (tiny).

FWS-side relevance: this is the **Functa pattern** (Dupont+ 2022) that
motivates the v3 hypernet ablations. The premise is that the renderable
distribution of weight tensors shares a low-dimensional manifold; the
shared body captures the manifold, the per-leaf FiLM picks the point on
it. Compare against pattern 1 (one INR per leaf, no sharing) and pattern
3 (a hypernet emits the shared body's params from external conditioning).
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
from ondes import SIREN

import loom


HIDDEN_DIM = 8
NUM_HIDDEN_LAYERS = 2


class TinyMLP(eqx.Module):
    """Same target as example 01 — keeps the comparison apples-to-apples."""

    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, *, key):
        k1, k2 = jax.random.split(key)
        self.fc1 = eqx.nn.Linear(in_features=4, out_features=6, key=k1)
        self.fc2 = eqx.nn.Linear(in_features=6, out_features=2, key=k2)

    def __call__(self, x):
        return self.fc2(jax.nn.relu(self.fc1(x)))


def make_coord_grid(shape: tuple[int, ...]) -> jax.Array:
    """Plain user-side coord grid; loom does not own this (Principle 5)."""
    axes = [jnp.linspace(-1.0, 1.0, n) if n > 1 else jnp.zeros(1) for n in shape]
    mesh = jnp.meshgrid(*axes, indexing="ij")
    return jnp.stack([m.ravel() for m in mesh], axis=-1)


def main() -> None:
    key = jax.random.key(0)
    k_target, k_body = jax.random.split(key)

    target = TinyMLP(key=k_target)

    # `ndim > 0` skips 0-D scalars (e.g. learnable temperatures) — coordinate-
    # based INRs need at least one axis to build a grid against.
    def is_float(x):
        return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating) and x.ndim > 0

    renderable, passthrough = eqx.partition(target, is_float)

    # KeyPath rank varies between leaves (e.g. weight is 2-D, bias is 1-D),
    # so the shared body needs a fixed input dim. We use the maximum rank
    # across renderable leaves and pad coord grids with zeros — a user-side
    # convention, not a loom concern.
    leaves_with_paths = jax.tree_util.tree_leaves_with_path(renderable)
    in_dim = max(len(leaf.shape) for _, leaf in leaves_with_paths)

    body = SIREN(in_dim=in_dim, hidden_dim=HIDDEN_DIM, num_hidden_layers=NUM_HIDDEN_LAYERS, key=k_body)

    # FiLM shape per Body.trunk: ``(num_hidden_layers, 2 * hidden_dim)``.
    # One small FiLM tensor per renderable leaf, keyed by string tag.
    def tag_of(path):
        return "/".join(getattr(p, "name", str(p)) for p in path)

    films = {tag_of(p): jnp.zeros((NUM_HIDDEN_LAYERS, 2 * HIDDEN_DIM)) for p, _ in leaves_with_paths}

    def pad_coords(coords: jax.Array, target_in_dim: int) -> jax.Array:
        if coords.shape[-1] == target_in_dim:
            return coords
        pad = jnp.zeros((coords.shape[0], target_in_dim - coords.shape[-1]))
        return jnp.concatenate([coords, pad], axis=-1)

    def f(path, shape, dtype, params):
        body, films = params
        coords = pad_coords(make_coord_grid(shape), in_dim)
        film = films[tag_of(path)]
        flat = jax.vmap(lambda c: body(c, film=film))(coords).astype(dtype)
        return flat.reshape(shape)

    rendered_renderable = loom.render(renderable, f, (body, films))
    rendered = eqx.combine(rendered_renderable, passthrough)

    assert rendered.fc1.weight.shape == target.fc1.weight.shape
    x = jnp.ones(4)
    print(f"forward-pass output: {rendered(x)}")

    # The whole point: shared body is large, FiLM is small. Show it.
    body_param_count = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(body, eqx.is_array)))
    film_param_count = sum(x.size for x in jax.tree_util.tree_leaves(films))
    print(f"shared body param count: {body_param_count}")
    print(f"total per-leaf FiLM param count ({len(films)} leaves): {film_param_count}")
    print(f"per-leaf FiLM param count (mean): {film_param_count // len(films)}")
    print("→ adding a new renderable leaf grows params by the FiLM size only, not by a fresh body.")


if __name__ == "__main__":
    main()
