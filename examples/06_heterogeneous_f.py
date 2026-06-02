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

import equinox as eqx
import jax
import jax.numpy as jnp
import ondes

import loom


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

    # Track which branch fires per path — for printing only.
    branch_log: dict[str, str] = {}

    def f(path, shape, dtype, params):
        conv_p, fc_p = params
        # Path is a tuple of jax KeyPath key-parts. eqx.Module attribute
        # access shows up as GetAttrKey(name='conv1'); we identify conv vs
        # fc by checking whether any attribute key starts with "conv".
        path_str = jax.tree_util.keystr(path)
        is_conv = "conv" in path_str
        branch_log[path_str] = "conv" if is_conv else "fc"

        n = 1
        for d in shape:
            n *= d
        coords_1d = jnp.linspace(-1.0, 1.0, n)[:, None]

        if is_conv:
            (conv_enc, conv_body_m) = conv_p
            encoded = jax.vmap(conv_enc)(coords_1d)
            ys = jax.vmap(conv_body_m)(encoded)
        else:
            ys = jax.vmap(fc_p)(coords_1d)
        return ys.reshape(shape).astype(dtype)

    params = ((conv_encoding, conv_body), fc_body)
    rendered = loom.render(renderable, f, params)

    # Recombine with non-float passthrough leaves so we get a full module.
    final = eqx.combine(rendered, passthrough)

    # Report
    print("Heterogeneous `f` rendered a CNN target with two parameterisations:")
    conv_paths = [p for p, b in branch_log.items() if b == "conv"]
    fc_paths = [p for p, b in branch_log.items() if b == "fc"]
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
