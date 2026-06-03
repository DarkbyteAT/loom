"""Pattern 6: heterogeneous `f` — different renderers per leaf group.

Real networks have weights with very different inductive biases at different
layers. Conv kernels are spatially smooth; fully-connected weights have no
spatial structure to exploit. Reasonable people use different parameterisations
for each — for example, a Fourier-feature-encoded SIREN for conv kernels
(exploiting locality) and a plain SIREN for FC layers (no encoding, treat
each weight as a 1-D coord).

The contract relevance:

- The dispatch happens *inside `f`*. loom does not own the dispatch; there
  is no `loom.register_renderer(path_pattern, renderer)` and there never
  will be. The user's `f` is the single point of dispatch (Principle 5).

- Heterogeneity does NOT require a parallel meta-tree. The native KeyPath
  (Guarantee — Principle 3) is the only identity carried; `f` can pattern-
  match on it however it likes (string match, prefix tuple, a stable
  tag_of_path dict — see PHILOSOPHY §"Refactor-safe dispatch").

- Both branches of the dispatch return the contract-correct shape and
  dtype; loom validates both branches transparently.

- `params` is a tuple `(conv_params, fc_params)`. Gradients flow back into
  whichever sub-tree the dispatch read from — autograd handles unused
  branches as zero contributions, so there's no double-gradient pitfall.

This is the "conv-and-FC" or "conv-and-attention" case downstream models
will need.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import ondes

import loom


def _attr_starts_with(path: tuple, prefix: str) -> bool:
    """True if any KeyPath part is a `GetAttrKey` whose name starts with `prefix`.

    Pattern-matching on `GetAttrKey.name` is more robust than substring-checking
    the rendered key-string — it can't false-positive on a dict key or sequence
    index that happens to contain the prefix.
    """
    return any(isinstance(k, jax.tree_util.GetAttrKey) and k.name.startswith(prefix) for k in path)


def _is_conv_path(path: tuple) -> bool:
    """Conv-branch predicate: any attribute key starts with "conv"."""
    return _attr_starts_with(path, "conv")


def _is_fc_path(path: tuple) -> bool:
    """FC-branch predicate: any attribute key starts with "fc"."""
    return _attr_starts_with(path, "fc")


class TinyConvNet(eqx.Module):
    """Conv -> Conv -> Flatten -> FC -> FC.

    Deliberately mixes two leaf-group categories (conv kernels and FC
    weight matrices) so the heterogeneous dispatch fires both branches.
    """

    conv1: eqx.nn.Conv2d
    conv2: eqx.nn.Conv2d
    fc1: eqx.nn.Linear
    fc2: eqx.nn.Linear

    def __init__(self, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.conv1 = eqx.nn.Conv2d(1, 4, kernel_size=3, key=k1)
        self.conv2 = eqx.nn.Conv2d(4, 8, kernel_size=3, key=k2)
        self.fc1 = eqx.nn.Linear(8, 16, key=k3)
        self.fc2 = eqx.nn.Linear(16, 10, key=k4)


def main() -> None:
    seed = jax.random.PRNGKey(11)
    k_target, k_conv_body, k_fc_body, k_conv_enc = jax.random.split(seed, 4)

    target = TinyConvNet(k_target)

    def is_float(x):
        return eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)

    renderable, passthrough = eqx.partition(target, is_float)

    # Two different renderers, each picked for the inductive bias of its
    # leaf group:
    #
    # - Conv kernels: encode 1-D coords with a Fourier feature encoding
    #   before the SIREN body. Gaussian encoding here is just to show
    #   "different machinery in this branch" — a real conv prior would
    #   use 2-D coords and exploit spatial locality.
    #
    # - FC weights: plain SIREN on raw 1-D coords. No encoding.
    conv_encoding = ondes.Gaussian(rank=1, num_freqs=8, sigma=3.0, key=k_conv_enc)
    conv_body = ondes.SIREN(in_dim=16, hidden_dim=16, num_hidden_layers=2, key=k_conv_body)
    fc_body = ondes.SIREN(in_dim=1, hidden_dim=16, num_hidden_layers=2, key=k_fc_body)

    # The renderer is *pure*: branch dispatch reads only `path` and `params`,
    # writes nothing to enclosing scope. loom's Guarantee 7 says iteration
    # order is unspecified and `f` may be traced/transformed, so any side
    # effect inside `f` is undefined behaviour. Branch tracking for the
    # report is recovered separately from `renderable`'s paths below.
    #
    # Dispatch is *exhaustive*: each predicate is explicit and any unmapped
    # path raises `KeyError`. Silent fallback to a default renderer would
    # mask architecture-evolution bugs (a new layer type added downstream
    # would get the wrong parameterisation without anyone noticing).
    def f(path, shape, dtype, params):
        conv_p, fc_p = params
        n = math.prod(shape)
        coords_1d = jnp.linspace(-1.0, 1.0, n)[:, None]

        if _is_conv_path(path):
            (conv_enc, conv_body_m) = conv_p
            encoded = jax.vmap(conv_enc)(coords_1d)
            ys = jax.vmap(conv_body_m)(encoded)
        elif _is_fc_path(path):
            ys = jax.vmap(fc_p)(coords_1d)
        else:
            raise KeyError(
                f"no renderer for path {jax.tree_util.keystr(path)!r}; "
                "add a branch in `f` (conv/fc are the only mapped families)"
            )
        return ys.reshape(shape).astype(dtype)

    params = ((conv_encoding, conv_body), fc_body)
    rendered = loom.render(renderable, f, params)

    # Recombine with non-float passthrough leaves so we get a full module.
    final = eqx.combine(rendered, passthrough)

    # Report — recover the per-branch partition from `renderable`'s paths.
    # No side effects from `f` needed.
    print("Heterogeneous `f` rendered a CNN target with two parameterisations:")
    conv_paths: list[str] = []
    fc_paths: list[str] = []
    for path, leaf in jax.tree_util.tree_leaves_with_path(renderable, is_leaf=eqx.is_array):
        if not eqx.is_array(leaf):
            continue
        # Mirror the dispatch in `f`: each leaf must hit exactly one branch.
        # If render() succeeded above, no leaf is unmapped here either.
        if _is_conv_path(path):
            conv_paths.append(jax.tree_util.keystr(path))
        elif _is_fc_path(path):
            fc_paths.append(jax.tree_util.keystr(path))
    print(f"  conv-branch leaves ({len(conv_paths)}):")
    for p in conv_paths:
        print(f"    {p}")
    print(f"  fc-branch leaves ({len(fc_paths)}):")
    for p in fc_paths:
        print(f"    {p}")
    assert len(conv_paths) > 0 and len(fc_paths) > 0, (
        "Heterogeneous dispatch needs to fire both branches to be a meaningful demo."
    )

    # Sanity: structure preservation (Guarantee 1) and shape contract
    # (Guarantee 2) held — loom would have raised otherwise.
    print("\nForward pass on a dummy input to confirm shapes are valid:")
    dummy = jnp.zeros((1, 7, 7))
    h = final.conv1(dummy)
    h = final.conv2(h)
    h = h.reshape(-1)[:8]
    h = final.fc1(h)
    h = final.fc2(h)
    print(f"  final logits shape: {h.shape} (expected (10,))")
    assert h.shape == (10,)
    print("Heterogeneous f works — both dispatch branches contributed contract-valid leaves.")


if __name__ == "__main__":
    main()
    print("PASS")
