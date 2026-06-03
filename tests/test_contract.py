"""Verify the seven contract guarantees from `docs/PHILOSOPHY.md`.

One test per guarantee, plus a wrapping-test for `RenderError`. The fixture
pytree is a minimal `eqx.Module` with two shape-bearing leaves of different
shapes and one static string field that should pass through.
"""

from __future__ import annotations

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from jaxtyping import Array

import loom
from loom import DTypeMismatch, RenderError, ShapeMismatch


pytestmark = pytest.mark.unit


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


# --- Defensive: non-array returns from `f` are wrapped, not raised raw -----


@pytest.mark.parametrize(
    ("value", "type_name"),
    [
        (None, "NoneType"),
        (42, "int"),
        ([1.0, 2.0, 3.0], "list"),
        ({"a": 1.0}, "dict"),
        ((1.0, 2.0), "tuple"),
    ],
)
def test_non_array_return_raises_render_error(value: Any, type_name: str):
    # Given: a renderer that returns a non-array Python value
    target = make_target()

    def bad(path, shape, dtype, params):
        return value

    # When/Then: the bad return is wrapped in RenderError with path + type context
    with pytest.raises(RenderError) as exc_info:
        loom.render(target, bad, None)

    msg = str(exc_info.value)
    assert (".w" in msg) or (".b" in msg)  # path-pointing
    # Subclass guarantees are not falsely triggered — this is the base class
    assert not isinstance(exc_info.value, ShapeMismatch)
    assert not isinstance(exc_info.value, DTypeMismatch)
    # The underlying TypeError carries the offending type name for the user
    cause = exc_info.value.__cause__
    assert isinstance(cause, TypeError)
    assert type_name in str(cause)


# --- Serialisation: error hierarchy survives cross-process round-trip -----


def test_render_errors_survive_cross_process_serialisation():
    """Exceptions must round-trip via the standard reconstruction protocol so
    JAX worker -> driver re-raise (multi-host / multiprocessing) works.
    """
    import pickle  # noqa: S403 -- exception reconstruction, not data deserialisation

    # Given: one instance of each error type with realistic context
    path = (jax.tree_util.GetAttrKey("w"),)
    base = RenderError(path, (3, 4), jnp.float32)
    shape_err = ShapeMismatch(path, (3, 4), (7,), jnp.float32)
    dtype_err = DTypeMismatch(path, (3, 4), jnp.float32, jnp.float64)

    # When: each is round-tripped through the serialisation protocol
    for original in (base, shape_err, dtype_err):
        revived = pickle.loads(pickle.dumps(original))  # noqa: S301 -- round-trip of our own class

        # Then: type, attributes, and rendered message all survive
        assert type(revived) is type(original)
        assert revived.path == original.path
        assert revived.shape == original.shape
        assert revived.dtype == original.dtype
        assert str(revived) == str(original)

    # And: subclass-specific attributes survive too
    revived_shape = pickle.loads(pickle.dumps(shape_err))  # noqa: S301
    assert isinstance(revived_shape, ShapeMismatch)
    assert revived_shape.actual_shape == (7,)

    revived_dtype = pickle.loads(pickle.dumps(dtype_err))  # noqa: S301
    assert isinstance(revived_dtype, DTypeMismatch)
    assert revived_dtype.actual_dtype == jnp.float64


# --- P0 gap fills (from final-pass QA review) ----------------------------


def test_dtype_mismatch_with_x64_disabled():
    """Default JAX config has x64 disabled — fp64 silently downgrades.

    The earlier dtype-negative test toggles `jax_enable_x64=True` to make
    the promotion fire. Most users will hit the default config, where the
    natural footgun is a dtype mismatch via a non-fp64 path. We verify
    DTypeMismatch fires for the `bf16 returned but fp32 expected` case,
    which is not subject to silent promotion.
    """
    # Given: a target whose leaves are fp32, and a renderer returning bf16
    target = make_target()

    def bf16_renderer(path, shape, dtype, params):
        return jnp.ones(shape, dtype=jnp.bfloat16)

    # When/Then: rendering raises DTypeMismatch under default JAX config,
    # with structured attrs (expected/actual dtype) populated — not just
    # the message string.
    with pytest.raises(DTypeMismatch) as exc_info:
        loom.render(target, bf16_renderer, None)
    err = exc_info.value
    assert isinstance(err, RenderError)
    assert "dtype" in str(err).lower()
    assert err.dtype == jnp.float32  # expected (P's leaf dtype)
    assert err.actual_dtype == jnp.bfloat16  # what f returned


# Note: pytree return cases (dict, tuple) are covered by
# `test_non_array_return_raises_render_error` above (parametrized).


def test_jit_without_static_argnums_documents_behaviour():
    """Characterise `jax.jit(render)` without `static_argnums=(1,)`.

    `f` is a Python callable — not a JAX array. JAX's jit asks the user to
    mark non-array positional arguments static. Calling `jax.jit(render)`
    without `static_argnums=(1,)` fails loudly at call time with a
    `TypeError` whose message names `static_argnums`/`static_argnames`,
    directing the user to the documented canonical pattern.

    The contract here is "fail loudly", not "auto-promote f to static":
    silent promotion would corrupt the jit cache across distinct renderers.
    """
    target = make_target()
    jitted_no_static = jax.jit(loom.render)

    # When/Then: calling without static treatment of `f` raises TypeError
    with pytest.raises(TypeError) as exc_info:
        jitted_no_static(target, ones_renderer, jnp.float32(1.0))
    msg = str(exc_info.value)
    # Error must mention the static-args remedy so users know the fix
    assert "static_argnums" in msg or "static_argnames" in msg

    # And: the documented canonical pattern works
    out = jax.jit(loom.render, static_argnums=(1,))(target, ones_renderer, jnp.float32(2.0))
    assert jnp.allclose(out.w, 2.0)


def test_g6_retraces_on_P_structure_change():
    """Positive complement to `test_jit_stability_no_retrace_on_param_change`.

    Guarantee 6 says re-jitting IS triggered by P-structure change. The
    earlier test asserts no-retrace on params change; this asserts retrace
    on P-structure change. Together they make G6 bidirectionally tested.
    """
    target_a = make_target()  # 2 shape-bearing leaves
    target_b = Target(
        w=jnp.zeros((6, 4), dtype=jnp.float32),  # different shape
        b=jnp.zeros((5,), dtype=jnp.float32),
        tag="hello",
    )
    trace_count = 0

    def counting_renderer(path, shape, dtype, params):
        nonlocal trace_count
        trace_count += 1
        return params * jnp.ones(shape, dtype=dtype)

    jitted = jax.jit(loom.render, static_argnums=(1,))

    # When: we render two targets of different leaf shapes (same tree count, different cache key)
    jitted(target_a, counting_renderer, jnp.float32(1.0))
    after_a = trace_count
    jitted(target_b, counting_renderer, jnp.float32(1.0))
    after_b = trace_count

    # Then: the second call re-traced (counter incremented further)
    assert after_b > after_a, f"expected retrace on P shape change; counter stayed at {after_a} -> {after_b}"


def test_render_inside_jax_lax_scan():
    """`render` composes with `jax.lax.scan` (canonical recipe from PHILOSOPHY)."""
    target = make_target()

    def step(params, _):
        rendered = loom.render(target, ones_renderer, params)
        return params + 1.0, rendered.w.sum()

    _, sums = jax.lax.scan(step, jnp.float32(0.0), jnp.arange(3))
    # sums[k] = sum(ones(3,4)) * k = 12 * k, so sums = [0, 12, 24]
    assert jnp.allclose(sums, jnp.array([0.0, 12.0, 24.0], dtype=jnp.float32))


def test_eqx_partition_selective_rendering_recipe():
    """The canonical selective-rendering recipe from PHILOSOPHY §"Composition recipes".

    Verifies `eqx.partition(target, is_float_array) → render → eqx.combine`
    works end-to-end: non-array leaves pass through, array leaves render.
    """

    def is_float_array(x: Any) -> bool:
        return bool(eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating))

    target = make_target()
    renderable, passthrough = eqx.partition(target, is_float_array)
    rendered_part = loom.render(renderable, ones_renderer, jnp.float32(2.0))
    final = eqx.combine(rendered_part, passthrough)

    # Then: shape-bearing leaves are rendered to 2 * ones; static `tag` passes through
    assert jnp.allclose(final.w, 2.0)
    assert jnp.allclose(final.b, 2.0)
    assert final.tag == "hello"


def test_grad_of_grad_through_render():
    """Second-order autodiff through `render` (Guarantee 5 corollary)."""
    target = make_target()

    def loss_fn(params):
        rendered = loom.render(target, lambda p, s, d, x: x * x * jnp.ones(s, d), params)
        return rendered.w.sum()

    # First derivative w.r.t. scalar params: d/dx (x^2 * 12) = 24 * x
    # Second derivative: d/dx (24 * x) = 24
    second = jax.grad(jax.grad(loss_fn))(jnp.float32(3.0))
    assert jnp.isfinite(second)
    assert float(second) == pytest.approx(24.0)


def test_mixed_dtype_P_renders_per_leaf_dtype():
    """`P` with multiple leaf dtypes — each leaf renders with its own dtype.

    Guarantee 3 is per-leaf; the existing dtype tests use a uniform-fp32 P.
    This exercises the mixed case to catch cross-leaf dtype leakage.
    """

    class MixedTarget(eqx.Module):
        f: Array
        i: Array

    mixed = MixedTarget(
        f=jnp.zeros((3,), dtype=jnp.float32),
        i=jnp.zeros((2,), dtype=jnp.int32),
    )

    def per_leaf_dtype(path, shape, dtype, params):
        # Honour the dtype loom hands us per-leaf
        return jnp.ones(shape, dtype=dtype)

    rendered = loom.render(mixed, per_leaf_dtype, None)
    assert rendered.f.dtype == jnp.float32
    assert rendered.i.dtype == jnp.int32
    assert jnp.allclose(rendered.f, 1.0)
    assert jnp.all(rendered.i == 1)
