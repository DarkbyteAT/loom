"""Verify the seven contract guarantees from `docs/PHILOSOPHY.md`.

One test per guarantee, plus a wrapping-test for `RenderError`. The fixture
pytree is a minimal `eqx.Module` with two shape-bearing leaves of different
shapes and one static string field that should pass through.
"""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

import loom
from loom import DTypeMismatch, RenderError, ShapeMismatch


class Target(eqx.Module):
    """Minimal synthetic pytree: two arrays + one static, non-shape leaf."""

    w: Array
    b: Array
    tag: str = eqx.field(static=True)


def make_target() -> Target:
    return Target(
        w=jnp.zeros((3, 4), dtype=jnp.float32),
        b=jnp.zeros((5,), dtype=jnp.float32),
        tag="hello",
    )


def ones_renderer(path, shape, dtype, params):
    """Pure renderer: returns `params * jnp.ones(shape, dtype)`."""
    return params * jnp.ones(shape, dtype=dtype)


# --- Guarantee 1: structure preservation ---------------------------------


def test_structure_preservation():
    # Given: a target pytree with two array leaves and one static field
    target = make_target()
    params = jnp.float32(2.0)

    # When: we render
    rendered = loom.render(target, ones_renderer, params)

    # Then: the tree structure is identical
    assert jax.tree_util.tree_structure(rendered) == jax.tree_util.tree_structure(target)
    assert rendered.tag == "hello"  # static field passed through


# --- Guarantee 2: shape correctness --------------------------------------


def test_shape_correctness_positive():
    # Given: a renderer that returns arrays of the expected shape
    target = make_target()
    params = jnp.float32(1.0)

    # When: we render
    rendered = loom.render(target, ones_renderer, params)

    # Then: each rendered leaf has the matching shape
    assert rendered.w.shape == (3, 4)
    assert rendered.b.shape == (5,)


def test_shape_correctness_negative_raises_shape_mismatch():
    # Given: a renderer that returns the wrong shape
    target = make_target()

    def bad(path, shape, dtype, params):
        return jnp.zeros((7,), dtype=dtype)  # wrong: w is (3, 4), b is (5,)

    # When/Then: rendering raises ShapeMismatch with path context
    with pytest.raises(ShapeMismatch) as exc_info:
        loom.render(target, bad, None)

    msg = str(exc_info.value)
    assert "shape" in msg.lower()
    # The first renderable leaf JAX visits will be either `.w` or `.b`; both
    # are acceptable — the contract is path-pointing, not order-fixed.
    assert (".w" in msg) or (".b" in msg)
    assert isinstance(exc_info.value, RenderError)  # subclass of base


# --- Guarantee 3: dtype rule ---------------------------------------------


def test_dtype_correctness_positive():
    # Given: a renderer that preserves dtype
    target = make_target()

    # When: we render
    rendered = loom.render(target, ones_renderer, jnp.float32(1.0))

    # Then: dtypes match
    assert rendered.w.dtype == jnp.float32
    assert rendered.b.dtype == jnp.float32


def test_dtype_negative_raises_dtype_mismatch():
    # Given: a renderer that promotes to float64 (the silent footgun)
    target = make_target()

    def promoting(path, shape, dtype, params):
        # Build an fp64 array regardless of `dtype` requested
        return jnp.ones(shape, dtype=jnp.float64)

    # When/Then: rendering raises DTypeMismatch (loom validates, does not cast)
    from jax import config

    config.update("jax_enable_x64", True)
    try:
        with pytest.raises(DTypeMismatch) as exc_info:
            loom.render(target, promoting, None)
        msg = str(exc_info.value)
        assert "dtype" in msg.lower()
        assert isinstance(exc_info.value, RenderError)
    finally:
        config.update("jax_enable_x64", False)


# --- Guarantee 4: vmap broadcast pattern ---------------------------------


def test_vmap_pattern_over_params():
    # Given: a batch of params (B=4), and the canonical in_axes spec
    target = make_target()
    params_batch = jnp.arange(4, dtype=jnp.float32)  # [0, 1, 2, 3]

    # When: we vmap render over the batch axis of params
    batched = jax.vmap(loom.render, in_axes=(None, None, 0))(target, ones_renderer, params_batch)

    # Then: each rendered leaf gains a leading batch axis
    assert batched.w.shape == (4, 3, 4)
    assert batched.b.shape == (4, 5)
    # And the batch axis multiplies through (params=[0,1,2,3], renderer = params * ones)
    assert jnp.allclose(batched.w[0], 0.0)
    assert jnp.allclose(batched.w[3], 3.0)


# --- Guarantee 5: gradient flow ------------------------------------------


def test_gradient_flow():
    # Given: a loss that sums all rendered values; params is a scalar
    target = make_target()

    def loss_fn(params):
        rendered = loom.render(target, ones_renderer, params)
        return rendered.w.sum() + rendered.b.sum()

    # When: we take the gradient
    grad = jax.grad(loss_fn)(jnp.float32(1.0))

    # Then: the gradient is finite and non-trivial
    # d(loss)/d(params) = sum(ones(3,4)) + sum(ones(5,)) = 12 + 5 = 17
    assert jnp.isfinite(grad)
    assert float(grad) == pytest.approx(17.0)


# --- Guarantee 6: JIT-stability ------------------------------------------


def test_jit_stability_no_retrace_on_param_change():
    # Given: a jitted render; we count traces via a side-channel counter
    target = make_target()
    trace_count = 0

    def counting_renderer(path, shape, dtype, params):
        nonlocal trace_count
        trace_count += 1
        return params * jnp.ones(shape, dtype=dtype)

    jitted = jax.jit(loom.render, static_argnums=(1,))

    # When: we call with two different param values of the same structure
    a = jitted(target, counting_renderer, jnp.float32(1.0))
    traces_after_first = trace_count
    b = jitted(target, counting_renderer, jnp.float32(2.0))
    traces_after_second = trace_count

    # Then: the second call did not re-trace (same P structure, same params shape)
    # Each trace calls the renderer once per leaf (2 leaves), so a retrace would
    # add 2 to the count.
    assert traces_after_first == traces_after_second, (
        f"expected no retrace; counter went {traces_after_first} -> {traces_after_second}"
    )
    # And both calls produced the expected scaled output
    assert jnp.allclose(a.w, 1.0)
    assert jnp.allclose(b.w, 2.0)


# --- Guarantee 7: iteration order unspecified ----------------------------


def test_iteration_order_unspecified():
    """Documented as unspecified — we do not assert any particular order.

    The guarantee defends against future contributors codifying an order that
    downstream code might come to depend on. A passing render under a pure
    renderer (which we already exercise above) is the only contract.
    """
    # Given: a pure renderer (no cross-call state)
    target = make_target()

    # When: we render twice
    a = loom.render(target, ones_renderer, jnp.float32(1.0))
    b = loom.render(target, ones_renderer, jnp.float32(1.0))

    # Then: outputs agree (any traversal order would yield the same result for
    # a pure f). We do NOT assert which leaf was visited first.
    assert jnp.allclose(a.w, b.w)
    assert jnp.allclose(a.b, b.b)


# --- Bonus: RenderError wraps arbitrary exceptions from f ----------------


def test_render_error_wraps_arbitrary_exceptions():
    # Given: a renderer that raises a non-loom exception (e.g. a dispatch miss)
    target = make_target()
    dispatch_table: dict[str, Array] = {}  # empty — will KeyError

    def dispatching(path, shape, dtype, params):
        return dispatch_table[str(path)]  # KeyError

    # When/Then: the KeyError is wrapped in RenderError with path context
    with pytest.raises(RenderError) as exc_info:
        loom.render(target, dispatching, None)

    msg = str(exc_info.value)
    assert "render failed" in msg.lower()
    assert (".w" in msg) or (".b" in msg)  # path-pointing
    assert isinstance(exc_info.value.__cause__, KeyError)  # original chained
    # And the subclass guarantees are not falsely triggered
    assert not isinstance(exc_info.value, ShapeMismatch)
    assert not isinstance(exc_info.value, DTypeMismatch)
