"""The `render` primitive — loom's single public entry point.

See `docs/PHILOSOPHY.md` for the seven contract guarantees. The implementation
is intentionally thin: `tree_map_with_path` carries structure preservation
(Guarantee 1), iteration order (Guarantee 7), and JIT-stability (Guarantee 6)
for us; we only contribute shape/dtype validation (Guarantees 2 and 3) and
error wrapping for exceptions raised by `f`.
"""

from __future__ import annotations

from typing import Any, Protocol

import jax
from jaxtyping import Array

from loom.errors import DTypeMismatch, RenderError, ShapeMismatch


class RenderFn(Protocol):
    """Signature of a renderer function.

    Called once per renderable leaf. Receives the leaf's `KeyPath` identity,
    its expected `shape` and `dtype`, and the user's `params` pytree. Must
    return a `jax.Array` of exactly `shape` and `dtype` — any deviation
    triggers `ShapeMismatch` or `DTypeMismatch`.
    """

    def __call__(
        self,
        path: tuple[Any, ...],
        shape: tuple[int, ...],
        dtype: Any,
        params: Any,
    ) -> Array:
        """Materialise one leaf."""
        ...


def render(P: Any, f: RenderFn, params: Any) -> Any:
    """Materialise `P` by evaluating `f` at every renderable leaf.

    For each leaf where `hasattr(leaf, 'shape')`, calls
    `f(path, leaf.shape, leaf.dtype, params)` and replaces the leaf with the
    returned array. Non-shape-bearing leaves pass through unchanged.

    Args:
        P: Target pytree (e.g. a model's weight tree). Defines the structure
            and the shape/dtype contract for each renderable leaf.
        f: Renderer function. See `RenderFn` for the contract.
        params: Parameters under optimisation. Threaded into `f` as its
            fourth argument; gradients flow back through it.

    Returns:
        A pytree with the same structure as `P`. Renderable leaves are
        replaced by `f`'s output; non-renderable leaves are preserved.

    Raises:
        ShapeMismatch: If `f` returns an array whose shape differs from
            the leaf's shape (Guarantee 2).
        DTypeMismatch: If `f` returns an array whose dtype differs from
            the leaf's dtype (Guarantee 3).
        RenderError: If `f` itself raises an exception — the original is
            chained via `__cause__`.
    """

    def render_leaf(path: tuple[Any, ...], leaf: Any) -> Any:
        if not hasattr(leaf, "shape") or not hasattr(leaf, "dtype"):
            return leaf
        expected_shape = tuple(leaf.shape)
        expected_dtype = leaf.dtype
        try:
            out = f(path, expected_shape, expected_dtype, params)
        except RenderError:
            raise
        except Exception as e:
            raise RenderError(path, expected_shape, expected_dtype) from e
        actual_shape = tuple(out.shape)
        if actual_shape != expected_shape:
            raise ShapeMismatch(path, expected_shape, actual_shape, expected_dtype)
        if out.dtype != expected_dtype:
            raise DTypeMismatch(path, expected_shape, expected_dtype, out.dtype)
        return out

    return jax.tree_util.tree_map_with_path(render_leaf, P)
