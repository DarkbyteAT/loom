"""Pattern 5: inner-loop adaptation through `loom.render`.

Reptile / iMAML / SAM all share a structural pattern: a fixed target P and
*adapting* params. The inner loop does K SGD steps on `task_loss(render(P, f, p))`,
producing an adapted `p*`. The outer loop differentiates *something* through
that inner trajectory (the precise something is what distinguishes the
algorithms — Reptile takes the displacement, iMAML uses the implicit function
theorem, SAM uses the ascent direction).

This example demonstrates the inner step. The outer is downstream — samgria
will live there.

The contract relevance:

- The static-target assumption (Guarantee 6) is exactly what makes this
  composable. `P` is the same across inner steps; only `params` changes,
  so `loom.render` retraces zero times across the scan.

- Gradients flow `task_loss → P_rendered → params` (Guarantee 5) without
  any loom-side machinery. `jax.grad(task_loss_of)` is plain autodiff
  through `render` — no "loom-aware optimiser" exists or is needed.

- The scan body is identical to PHILOSOPHY.md §"Test the substrate" case 5.
  If this pattern needed a special primitive, the substrate would have failed.

K is capped at 5 to keep the example fast and the trajectory plot readable.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import ondes

import loom


def make_tiny_target(key: jax.Array) -> eqx.Module:
    """Single-layer linear map — minimal renderable target."""
    return eqx.nn.Linear(3, 5, key=key)


def main() -> None:
    seed = jax.random.PRNGKey(7)
    k_target, k_body, k_task, k_x = jax.random.split(seed, 4)

    target = make_tiny_target(k_target)

    def is_float(x):
        return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)

    renderable, passthrough = eqx.partition(target, is_float)

    # The renderer: a single shared SIREN body that outputs scalars; we
    # query it at one coord per output element and reshape. This is the
    # most-minimal canonical case (Pattern 1's single-body variant).
    body_init = ondes.SIREN(in_dim=1, hidden_dim=16, num_hidden_layers=2, key=k_body)

    def f(path, shape, dtype, params):
        # `params` here *is* the body. The whole point: gradients flow back
        # into the body's weights via standard autodiff.
        n = math.prod(shape)
        coords = jnp.linspace(-1.0, 1.0, n)[:, None]
        ys = jax.vmap(params)(coords)
        return ys.reshape(shape).astype(dtype)

    # A synthetic task: render the target, run a linear forward pass on a
    # fixed input batch, fit a fixed target output. The "task" is arbitrary —
    # what matters is that its gradient flows back through `loom.render` to
    # the body params.
    x_batch = jax.random.normal(k_x, (8, 3))
    y_target = jax.random.normal(k_task, (8, 5))

    def task_loss(rendered_target: eqx.Module) -> jax.Array:
        full = eqx.combine(rendered_target, passthrough)
        y_pred = jax.vmap(full)(x_batch)
        return jnp.mean((y_pred - y_target) ** 2)

    def task_loss_of(p):
        return task_loss(loom.render(renderable, f, p))

    # Inner-loop SGD. Exactly the snippet in PHILOSOPHY §"Test the substrate".
    lr = 1e-2
    K = 5

    def inner_step(p, _):
        # `eqx.filter_value_and_grad` ignores non-array leaves (static
        # metadata, ints, bool flags) — drop-in safer than `jax.value_and_grad`
        # for downstream consumers whose modules carry static fields.
        loss, grads = eqx.filter_value_and_grad(task_loss_of)(p)
        # Pytree subtraction: walk both trees, subtract leaf-wise. Equinox
        # makes this read like ordinary arithmetic on float arrays only.
        p_new = jax.tree_util.tree_map(
            lambda a, g: a - lr * g if eqx.is_array(a) and eqx.is_array(g) else a,
            p,
            grads,
        )
        return p_new, loss

    _adapted, loss_trace = jax.lax.scan(inner_step, body_init, jnp.arange(K))

    # Report the trajectory. A monotonically decreasing trace confirms
    # gradients are actually flowing through render(P, f, params).
    print(f"Inner-loop SGD on task_loss(render(P, f, params)) for K={K} steps.")
    print("Loss trajectory (through-render — gradients pass through `loom.render`):")
    for k, loss_k in enumerate(loss_trace):
        print(f"  step[{k}]: loss = {float(loss_k):.6f}")
    print(f"\nFinal vs initial loss: {float(loss_trace[-1]):.6f} vs {float(loss_trace[0]):.6f}")
    assert float(loss_trace[-1]) < float(loss_trace[0]), (
        "Inner-loop adaptation didn't reduce loss — gradient flow is broken."
    )
    print("Loss decreased — gradient flow through `loom.render` confirmed (Guarantee 5).")

    # ------------------------------------------------------------------
    # Contrast smoke test: SGD directly on the rendered Linear weights.
    #
    # The through-render loop above adapts the SIREN body's parameters,
    # and `loom.render` materialises the Linear's weights each step.
    # The direct loop below skips `loom.render` entirely: it materialises
    # the Linear once at `body_init`, then takes K SGD steps on the
    # Linear's weight/bias arrays directly.
    #
    # Both trajectories start from the SAME initial rendered target, so
    # they are directly comparable. The claim under test is "render
    # doesn't break gradient flow" — i.e. both loops should reduce loss.
    # They are NOT expected to land at the same point: the through-render
    # loop is a more constrained optimisation (every Linear weight is a
    # function of the same shared body, so SIREN-body updates couple all
    # Linear weights together), while the direct loop has independent
    # degrees of freedom per Linear weight.
    #
    # This is a smoke test, not a baseline — see README. It establishes
    # that the substrate's gradient pathway is functional, not that
    # render is competitive with direct training.
    # ------------------------------------------------------------------
    target_rendered_init = loom.render(renderable, f, body_init)

    def task_loss_direct(rendered_p):
        return task_loss(rendered_p)

    def direct_step(p, _):
        loss, grads = eqx.filter_value_and_grad(task_loss_direct)(p)
        p_new = jax.tree_util.tree_map(
            lambda a, g: a - lr * g if eqx.is_array(a) and eqx.is_array(g) else a,
            p,
            grads,
        )
        return p_new, loss

    _adapted_direct, loss_trace_direct = jax.lax.scan(direct_step, target_rendered_init, jnp.arange(K))

    print("\nLoss trajectory (direct — SGD on rendered Linear weights, no render in loop):")
    for k, loss_k in enumerate(loss_trace_direct):
        print(f"  step[{k}]: loss = {float(loss_k):.6f}")

    print("\nContrast smoke test — both trajectories from the same initial rendered target:")
    print(f"  through-render: {'  '.join(f'{float(L):.4f}' for L in loss_trace)}")
    print(f"  direct:         {'  '.join(f'{float(L):.4f}' for L in loss_trace_direct)}")
    initial = float(loss_trace[0])
    final_through = float(loss_trace[-1])
    final_direct = float(loss_trace_direct[-1])
    print(f"  initial={initial:.4f}  final_through_render={final_through:.4f}  final_direct={final_direct:.4f}")
    assert final_through < initial, "through-render did not reduce loss"
    assert final_direct < initial, "direct SGD did not reduce loss"
    print(
        "Both decreased — render's gradient pathway is functional. "
        "(Trajectories are NOT expected to match: through-render couples all "
        "Linear weights through a shared SIREN body; direct has independent DoF.)"
    )


if __name__ == "__main__":
    main()
    print("PASS")
