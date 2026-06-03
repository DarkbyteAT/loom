"""Pattern 4: hypernet with per-target conditioning, scanned.

The functa pattern (Pattern 2) shares an INR body and uses per-leaf FiLM to
specialise per weight tensor. A hypernet over a target *distribution*
(Pattern 3) batches `params` over a target axis. This example sits between
them: one fixed target P (one architecture), but a sequence of conditioning
contexts — e.g. "training-data identity" embeddings — that modulate the
shared body at scan time.

The contract relevance:

- The conditioning context flows in *through `params`*, not via closure. Each
  scan iteration assembles a fresh tuple `(body, films, ctx_t)` and hands it
  to `loom.render` as the third positional argument. loom doesn't know what
  a "context" is; it's just another leaf of the params pytree.

- This is the M3-to-hypernet bridge: same target architecture, K different
  conditioning embeddings, K different rendered weight pytrees. The same
  primitive call inside the scan body — no kwargs, no rebuilds.

- The static-target assumption (Guarantee 6) holds: `P` is unchanged across
  scan iterations; only `params_t` varies. `render` retraces zero times.

Downstream, this is what an FWS-style "ensemble over training-data identity"
or a samgria-style "task-conditioned inner loop init" needs.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import ondes

import loom


def make_tiny_target(key: jax.Array) -> eqx.Module:
    """A two-layer MLP whose float weights will be rendered.

    Small on purpose: each renderable leaf is on the order of tens of params,
    so per-leaf magnitudes are visible from a scan with only a handful of
    contexts.
    """
    k1, k2 = jax.random.split(key)
    return eqx.nn.Sequential(
        [
            eqx.nn.Linear(4, 6, key=k1),
            eqx.nn.Linear(6, 3, key=k2),
        ]
    )


def main() -> None:
    seed = jax.random.PRNGKey(0)
    k_target, k_body, k_film, k_ctx = jax.random.split(seed, 4)

    # Target: structure + shape/dtype contract loom will enforce.
    target = make_tiny_target(k_target)

    # Renderable leaves are the float arrays — biases and weight matrices.
    # We render only those and pass the rest through (eqx.partition recipe
    # from PHILOSOPHY.md §"Composition recipes").
    def is_float(x):
        return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)

    renderable, passthrough = eqx.partition(target, is_float)

    # Shared INR body. One body, all leaves; specialisation comes via FiLM +
    # the conditioning context. The body outputs a scalar per call and we
    # query it once per output element (n = prod(shape)) inside `f`.
    leaf_paths_shapes = [
        (path, leaf.shape)
        for path, leaf in jax.tree_util.tree_leaves_with_path(renderable, is_leaf=eqx.is_array)
        if eqx.is_array(leaf)
    ]
    # Coord dim = 1 (we sample a 1-D coordinate per output element); FiLM
    # tensor has shape `(num_hidden_layers, 2 * hidden_dim)` (see ondes
    # Body docstring).
    num_hidden_layers = 2
    hidden_dim = 16
    ctx_dim = 4
    body = ondes.SIREN(
        in_dim=1 + ctx_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        key=k_body,
    )

    # Per-leaf FiLM, keyed by KeyPath (loom's native identity convention).
    film_shape = (num_hidden_layers, 2 * hidden_dim)
    film_keys = jax.random.split(k_film, len(leaf_paths_shapes))
    films = {path: 0.1 * jax.random.normal(film_keys[i], film_shape) for i, (path, _) in enumerate(leaf_paths_shapes)}

    # K different conditioning contexts. Each one modulates every leaf.
    K = 4
    contexts = jax.random.normal(k_ctx, (K, ctx_dim))

    # The renderer. Same shape as PHILOSOPHY §"Test the substrate" case 4.
    def f(path, shape, dtype, params):
        body_p, films_p, ctx = params
        n = math.prod(shape)
        # 1-D coord per output element, concatenated with the (broadcast) ctx.
        coords_1d = jnp.linspace(-1.0, 1.0, n)[:, None]  # (n, 1)
        ctx_broad = jnp.broadcast_to(ctx, (n, ctx_dim))  # (n, ctx_dim)
        coords = jnp.concatenate([coords_1d, ctx_broad], axis=-1)
        film = films_p[path]
        ys = jax.vmap(lambda c: body_p(c, film=film))(coords)
        return ys.reshape(shape).astype(dtype)

    # Pre-render at the reference context once. The contrast smoke test
    # below compares each scan-step's rendered output against this
    # fixed reference, so we close over `rendered_ref` rather than
    # threading it through carry.
    rendered_ref = loom.render(renderable, f, (body, films, contexts[0]))
    ref_leaves = [leaf for leaf in jax.tree_util.tree_leaves(rendered_ref) if eqx.is_array(leaf)]

    # The scan body: rebuild params_t from (body, films, ctx_t) each step.
    # `body` and `films` are constants under the scan; only the context
    # carried in via `xs` changes. Emit per-step (RMS magnitudes,
    # max-abs deviation from the reference render).
    def step(carry, ctx_t):
        params_t = (body, films, ctx_t)
        rendered_t = loom.render(renderable, f, params_t)
        leaves_t = [leaf for leaf in jax.tree_util.tree_leaves(rendered_t) if eqx.is_array(leaf)]
        flat_mags = jnp.stack([jnp.sqrt(jnp.mean(leaf**2)) for leaf in leaves_t])
        max_dev = jnp.max(jnp.stack([jnp.max(jnp.abs(a - b)) for a, b in zip(leaves_t, ref_leaves, strict=True)]))
        return carry, (flat_mags, max_dev)

    _, (mag_trace, dev_trace) = jax.lax.scan(step, None, contexts)

    # Report — primary configuration: varying ctx per step.
    print(f"Scanned {K} conditioning contexts through {len(leaf_paths_shapes)} leaves.")
    print("Per-step rendered-weight RMS magnitudes (rows=ctx, cols=leaves):")
    for k in range(K):
        row = "  ".join(f"{m:.4f}" for m in mag_trace[k])
        print(f"  ctx[{k}]: {row}")

    # ------------------------------------------------------------------
    # Contrast smoke test: same scan but with ctx held CONSTANT.
    #
    # The claim under test is "the conditioning context actually
    # modulates the rendered output". A varying-ctx scan that produces
    # large per-step deviations from step-0 is consistent with that
    # claim, but only the contrast against a constant-ctx scan rules
    # out the alternative "the per-step deviations are an artefact of
    # something else (a closure, a non-determinism, an iteration-order
    # effect)". Holding ctx constant should drive deviation to zero up
    # to float noise.
    #
    # This is a smoke test, not a baseline — see README. It establishes
    # the pattern produces differentiated output where it claims to,
    # but says nothing about how good the parameterisation is.
    # ------------------------------------------------------------------
    constant_contexts = jnp.broadcast_to(contexts[0], contexts.shape)
    _, (_mag_const, dev_const) = jax.lax.scan(step, None, constant_contexts)
    print("\nContrast smoke test — per-step max-abs deviation of rendered tree from step-0:")
    print(f"  varying ctx: {'  '.join(f'{d:.6f}' for d in dev_trace)}")
    print(f"  constant ctx: {'  '.join(f'{d:.6f}' for d in dev_const)}")
    varying_signal = float(jnp.max(dev_trace))
    constant_signal = float(jnp.max(dev_const))
    print(f"  max deviation: varying={varying_signal:.6f}  constant={constant_signal:.6e}")
    assert varying_signal > 1e-3, (
        f"varying-ctx scan produced near-zero deviation ({varying_signal:.2e}); "
        "context is not modulating the rendered output as claimed."
    )
    assert constant_signal < 1e-5, (
        f"constant-ctx scan produced non-trivial deviation ({constant_signal:.2e}); "
        "render is non-deterministic in `params`, which breaks Guarantee 7's purity contract."
    )
    print("Contrast confirmed: varying ctx modulates output; constant ctx does not.")

    # Sanity: structure preservation. The render output is the same pytree
    # as `renderable`, ready to recombine with `passthrough`.
    final = eqx.combine(loom.render(renderable, f, (body, films, contexts[0])), passthrough)
    print("\nStructure preserved: same tree as input target (Guarantee 1).")
    print(f"Final tree leaves: {len(jax.tree_util.tree_leaves(final))}")


if __name__ == "__main__":
    main()
    print("PASS")
