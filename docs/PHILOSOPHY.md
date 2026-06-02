# loom — Philosophy

## The question

What is loom for?

## The verdict

> **loom is a substrate for re-parameterising JAX pytrees by the output of arbitrary functions, in a way that is scannable and optimisable.**

That's it. Not a renderer for neural-network weights. Not a hypernet framework. Not a training studio. A *substrate*. The substrate is function-agnostic — INRs from `ondes` are the canonical instantiation, but loom does not bake them in. Anything that maps `(path, shape, dtype, params) → array-of-leaf-shape` is a valid renderer.

Everything else — target architectures, task configs, training loops, diagnostics, plotting, sweep registries, coord-grid conventions, distribution heads — lives downstream. loom owns the mechanism, not any specific instantiation of it.

The library is thin by design. The product is the **contract** (§"Contract guarantees"), not the lines of code.

## A note on `path`

Throughout this document, `path` refers to a `jax.tree_util.KeyPath` — JAX's canonical way of identifying a leaf by its location in a pytree. A path is a tuple of "key parts" like `(GetAttrKey('conv1'), GetAttrKey('weight'))`, meaning *the `.weight` attribute of the `.conv1` attribute*. The key part types JAX exposes are:

- `GetAttrKey('field')` for `eqx.Module` / dataclass attribute access
- `DictKey('key')` for dict-keyed access
- `SequenceKey(i)` for list / tuple index access
- `FlattenedIndexKey(i)` for the flattened fallback

You get one path per leaf when you call `jax.tree_util.tree_flatten_with_path(P)` or `tree_leaves_with_path(P)`. loom uses paths as the substrate's identity convention because they are the only leaf-identity primitive JAX itself ships — using anything else would force a parallel structure and violate Principle 3.

## Six principles

Each principle is followed by *what it rules out* — the negative space matters as much as the positive claim.

### 1. Simple over expressive

*Operationalised form:* **every named concept must appear in the signature of a primitive a user calls.** That kills `Slot`, `LeafConditioning`, `Condition`, `Renderer`, `Target` reflexively — none of them appear in `render(P, f, params)`.

**Rules out**: value-type proliferation, registry abstractions, configuration objects, anything that grows the conceptual surface without earning it.

### 2. Mathematically transparent

*Falsifiable form:* **if the maths says `W = f(meta, params)` for some leaf-identifying meta, the code reads `f(meta, params)` with nothing else in the signature.** Any deviation must be named in the docs as an explicit cost-paid.

The substrate signature is `f(path, shape, dtype, params) → array`. `path` is the leaf's identity (per §"A note on `path`"); `shape, dtype` describe the tensor `f` must return; `params` are the parameters under optimisation. All four appear in the maths or are named cost.

```python
# What the maths says — the user thinks in these terms:
# f_path: params → tensor of shape leaf.shape
# loom evaluates f_path(params) for each renderable leaf

# What the code reads:
loom.render(P, lambda path, shape, dtype, params: f_for(path).materialise(shape, params), params)
```

*Worked rejection.* Principle 2 *rejects* signature additions like:

```python
# REJECTED: passes runtime state through the signature instead of through params.
def render(P, f, params, *, training: bool, key: PRNGKey) -> PyTree: ...
# `training` and `key` are not in the maths; they would be hidden cost.
# The user threads them through params instead:
params = {"theta": theta, "training": training, "key": key}
```

**Rules out**: ergonomic-but-impure wrappers, helper layers that hide the mathematical structure for brevity, anything where the user can't tell which lines are doing the maths.

### 3. Minimally invasive on the user's pytree

The user has a target pytree `P` (e.g. a CNN's weight tree). loom re-parameterises it without forcing the user to restructure `P`, attach parallel metadata trees, or wrap leaves in marker types. The substrate works *on* the user's pytree as-is. Identity is the native JAX `KeyPath`; leaf shape and dtype come from `hasattr(leaf, 'shape')` duck-typing.

**Rules out**: parallel meta-trees, custom leaf-wrappers, `loom.Leaf(...)` marker types, `@loom.renderable` decorators, anything that asks the user to mutate their model definition. (The recipe in §"Refactor-safe dispatch" provides an escape hatch the user owns.)

### 4. Scan- and optimise-native

*Operationalised form:* `render`'s output is a pytree where `jax.tree_util.tree_structure(rendered) == jax.tree_util.tree_structure(P)` and all rendered leaves are `jax.Array` instances. Gradients flow `loss → P_rendered → params` via standard autodiff. `jax.vmap(render, in_axes=(None, None, 0))` over batched `params` works. `jax.lax.scan` with `render` in the body works.

*Static-target assumption:* `render` traces `P` once per JIT cache key. Calling `render` under `vmap` / `scan` requires `P`'s structure to be identical across the mapped axis — only `params` varies. Heterogeneous target structures need separate `render` calls. *Cross-axis target heterogeneity is not in scope.*

**Rules out**: custom gradient-flow APIs, "loom-aware" optimisers, anything that requires `jax.lax.scan(loom.scan_step, ...)` instead of plain `jax.lax.scan`.

### 5. Mechanism, not policy

loom does not interpret what the user's function *is*. It does not know whether `f` is an INR, a hypernetwork, a closed-form mapping, or a lookup table. It does not own coord-grid construction, distribution choices, conditioning conventions, or filtering policy. The user composes `ondes` (or anything else) into `f`; loom hands `f` the leaf's identity, shape, dtype, and params, takes back the value, and threads gradients.

*Cost-paid.* In the canonical INR-per-weight case, `f`'s body is `jax.vmap(inr)(coord_grid(shape)).reshape(shape)`. This three-line pattern lives in user code or in `ondes`, not in loom. The substrate is willing to pay this readability cost to avoid owning coord-grid policy.

**Rules out**: `loom.heads`, `loom.distributions`, `loom.basis`, `loom.coord_grid`, `loom.nyquist_*`, anything that ships opinionated primitives for what `f` should look like. ondes ships those (where applicable). loom doesn't re-export them.

### 6. Composable, not configurable

When something needs to vary — coord-grid choice, filtering, multi-target batching, memory checkpointing — the answer is *user composes loom with JAX primitives + their own `f`*, not *loom grows a kwarg or a flag*. See §"Composition recipes" for the canonical patterns.

**Rules out**: future kwarg-bloat. Every PR proposing a new `render(..., mode='X')` flag has to first show why composition with `jax.checkpoint` / `jax.vmap` / `jax.lax.scan` / `eqx.partition` / user-side dispatch can't express it.

## The primitive

loom has **one primitive**:

```python
def render(
    P: PyTree,
    f: Callable[[KeyPath, Shape, DType, PyTree], Array],
    params: PyTree,
) -> PyTree:
    """Materialise the pytree.

    For each leaf at `path` in `P` where `hasattr(leaf, 'shape')`:
        rendered_leaf = f(path, leaf.shape, leaf.dtype, params)
        # invariant: rendered_leaf.shape == leaf.shape    (Guarantee 2)
        # invariant: rendered_leaf.dtype == leaf.dtype    (Guarantee 3 — validated)

    For non-shape-bearing leaves: pass through unchanged.

    Returns a pytree with the same structure as P; renderable leaves
    replaced by f's output, others preserved. f MUST be a pure function
    of its arguments (Guarantee 7 — iteration order is unspecified).

    Gradient flows from rendered through f's `params` argument into
    `params` via standard JAX autodiff. f may also close over additional
    state; gradients flow into those closures too, but loom does not
    validate that the closure matches `params`.
    """
```

That is the entire public surface. Three positional args, no kwargs. The positional ABI `(P, f, params)` is the stable API — future additions, if any, only come after `params`.

## Composition recipes

The patterns that earlier drafts considered as second primitives are JAX-primitive compositions:

```python
# Memory-efficient backward (addressed via jax.checkpoint, not a loom primitive)
loss = jax.checkpoint(lambda p: model(loom.render(P, f, p), x, y))(params)

# Batched targets / hypernet over a distribution
rendered_batch = jax.vmap(loom.render, in_axes=(None, None, 0))(P, f, params_batch)

# Inner-loop adaptation (e.g. iMAML / SAM / Reptile inner step)
def task_loss_of(p):
    return task_loss(loom.render(P, f, p))

def inner_step(p, _):
    return p - lr * jax.grad(task_loss_of)(p), None

adapted, _ = jax.lax.scan(inner_step, params, jnp.arange(K))

# Selective rendering — float-only filter (the canonical case;
# users with batchnorm stats / int leaves should default to this)
import equinox as eqx
import jax.numpy as jnp

is_float_array = lambda x: eqx.is_array(x) and jnp.issubdtype(x.dtype, jnp.floating)
renderable, passthrough = eqx.partition(target, is_float_array)
rendered = loom.render(renderable, f, params)
final = eqx.combine(rendered, passthrough)
```

If a recipe takes more than five lines to express, *that's* the gap to consider closing — not by adding to loom, but by adding to whichever sibling library is responsible (ondes for INR composition, fws for diagnostics, downstream for studio behaviour).

## Contract guarantees

The product loom sells is not the implementation — it's these guarantees:

1. **Structure preservation.** `tree_structure(render(P, f, params))` equals `tree_structure(P)`. Renderable leaves are replaced; non-shape-bearing leaves pass through unchanged.
2. **Shape correctness.** For every renderable leaf, the rendered output has `.shape == leaf.shape`. Loom raises `loom.ShapeMismatch` (a `loom.RenderError` subclass) with a path-pointing error message when `f` violates this.
3. **dtype rule.** Rendered output dtype must equal `P`'s leaf dtype. Loom does **not** cast — it *validates* and raises `loom.DTypeMismatch` (also a `loom.RenderError` subclass) on mismatch. This preserves mechanism-not-policy while killing the silent fp32 → fp64 promotion footgun. The user casts inside `f` if they want the conversion.
4. **Vmap broadcast pattern.** `render(P, f, params)` treats `P` and `f` as static and `params` as the mappable argument under `jax.vmap`. The canonical batched-render call is `jax.vmap(render, in_axes=(None, None, 0))(P, f, params_batch)`. Mapping over `P` violates the static-target assumption; mapping over `f` is meaningless (closures don't vmap).
5. **Gradient flow.** Standard JAX autodiff applies. `params` flows into `f` as its fourth argument; gradients flow back through it. Loom does not intercept gradients.
6. **JIT-stability.** `render` is jittable. Re-jitting is triggered only by changes in `P`'s tree structure (not by changes in `params` values, leaf values, or any other arg that fits `P`'s static structure). The static-target assumption from Principle 4 is the load-bearing invariant.
7. **Iteration order is unspecified.** `render` MAY traverse renderable leaves in any order, including parallel evaluation under future backends. `f` MUST be a pure function of `(path, shape, dtype, params)`; any closure over mutable cross-call state is undefined behaviour. This guarantee defends the cross-leaf-coupling non-goal — without it, a future contributor could codify iteration order and downstream code could start depending on it.

When `f` itself raises an exception (e.g. a `KeyError` from a dispatch dict, an `AttributeError` from a closure), loom wraps it in `loom.RenderError` with the path, shape, dtype, and original exception attached. The hierarchy:

```
loom.RenderError                  # base; wraps any exception from f
├── loom.ShapeMismatch            # Guarantee 2 violation
└── loom.DTypeMismatch            # Guarantee 3 violation
```

These seven guarantees are testable; the test suite verifies them on a synthetic minimal pytree before any release.

## What loom owns

- `render` (the primitive above)
- The seven contract guarantees
- The `KeyPath` convention as the leaf-identity type
- The `loom.RenderError` exception family

## What loom does NOT own

- Coord-grid construction or conventions (Nyquist, linspace, dyadic, anything else) — `ondes` or user code
- Filtering / partitioning of which leaves to render — `eqx.partition` and user composition
- Lazy materialisation under `jit` / `scan` / `grad` — structurally impossible (JAX traces materialise at trace time); eager-Python lazy is achievable only by violating Guarantee 1, so it does not live in the substrate
- Memory-efficient backward — `jax.checkpoint` composition
- Multi-target supervision primitives — `jax.vmap(render, ...)` composition
- Inner-loop meta-learning primitives — `jax.lax.scan` composition
- Concrete `f` implementations (INRs, hypernets, etc.) — `ondes` or downstream
- Target architectures, task configs, data loaders, training loops — downstream
- `Condition`, `Target`, `TaskCfg`, `RunResult`, `Renderer` value types — downstream or unnecessary
- FiLM modulation, polar orthogonalisation, basis composition — `ondes` or user code
- Sweep registries, plotting, diagnostics — downstream (`fws` for FWS programme)
- Distribution heads, Bayesian wrappers, ELBO machinery — downstream
- `default_coord_grid`, `nyquist_grid`, or any built-in coord builder — downstream (`ondes`)
- `should_render` predicate or any partition policy — `eqx.partition`

## Non-goals (explicit)

The principles above rule each of these out structurally. This section restates them as concrete anti-features for future PR reviewers — it's the explicit checklist a reviewer walks when someone proposes adding any of them back. There is *intentional* overlap with the principles and the "what loom does NOT own" list.

- **Output-coupled rendering.** No leaf's `f` call observes another leaf's *rendered output*. Coupling-via-`params` (shared parameters whose gradients touch jointly) IS supported and is the intended channel for cross-leaf structure (for example, FWS-style spectral priors imposed via shared body params). Coupling-via-output (leaf A's render fed into leaf B's `f`) is out of scope. Loom does not ship `render_sequential`, `render_with_dependencies`, or any DAG-aware variant.
- **Loss computation.** `render` returns a pytree. What you do with it (forward pass, loss, backprop) is downstream. No `render_and_loss`, no `render_and_grad`. The composition `jax.grad(loss ∘ render)` is one line; the substrate does not ship the composition.
- **Lazy materialisation inside JIT / scan / grad.** Structurally impossible per JAX trace semantics — JAX evaluates every leaf at trace time. Eager-Python lazy is achievable only by violating Guarantee 1 (returning Python callables instead of `jax.Array`), so it does not live in the substrate.
- **Heterogeneous target structures across `vmap` / `scan` axes.** The static-target assumption (Principle 4, Guarantee 6) is load-bearing.
- **Distribution / variational machinery.** `f` may stochastically sample (closing over a key, or threading the key through `params`); loom does not interpret stochasticity, KL terms, or ELBOs.

*Criterion for re-adding any of these.* A named consumer in the FWS / samgria / ondes / fws roadmap, with measured cost (an actual OOM, a measured memory pressure, an actual edge-inference deliverable). Speculation about transformers-as-targets, hypothetical streaming-inference, or generic "memory might be tight" does not clear this bar.

## Refactor-safe dispatch (recipe, not API)

The native identity is `KeyPath`. Paths are fragile under field renames — rename `conv1` → `conv_a` and every dispatch dict keyed on the old path breaks. The substrate accepts this cost in exchange for Principle 3 (no wrapper types). Users who need refactor-safety own a small adapter, keyed on the canonical path string produced by `jax.tree_util.keystr`:

```python
import jax.tree_util as jtu

# Stable tag map lives in user code, next to the model.
# Keys are jtu.keystr(path) strings — canonical across every JAX key type
# (GetAttrKey, DictKey, SequenceKey, FlattenedIndexKey).
tag_of_path = {
    ".conv1.weight": "conv_first",
    ".conv2.weight": "conv_second",
    ".head.weight":  "head",
}

def f(path, shape, dtype, params):
    key = jtu.keystr(path)
    tag = tag_of_path.get(key)
    if tag is None:
        raise KeyError(f"no tag for renderable path {key!r}; add it to tag_of_path")
    return inrs[tag].materialise(shape, dtype, params[tag])

rendered = loom.render(P, f, inr_params_by_tag)
```

Rename `conv1` → `conv_a`: update one line in `tag_of_path`, everything else holds. Five-line user pattern, not a loom primitive — preserves Principle 3 and gives the escape hatch.

**Whole-pytree vs selective tagging.** The recipe above assumes *whole-pytree tagging*: every leaf reaching `f` is renderable and must have a tag — a missing key is a user error and should fail loudly with the path attached. If you instead want *selective tagging* (some leaves rendered by INRs, others passed through unchanged), do not branch inside `f` on a missing tag — that mixes substrate concerns with model concerns. Use `eqx.partition` upstream to split the model into the renderable subtree (handed to `loom.render`) and the pass-through subtree (left alone), then `eqx.combine` the results. The substrate sees one homogeneous tree; `f` stays a total function over the leaves it receives.

**Why `keystr`, not `tuple(p.name for p in path if hasattr(p, 'name'))`?** JAX paths mix key types: `GetAttrKey` exposes `.name`, `DictKey` exposes `.key`, `SequenceKey` exposes `.idx`, `FlattenedIndexKey` exposes `.key`. The `hasattr(p, 'name')` filter silently drops every non-attribute key, collapsing distinct paths (e.g. `layers[0].weight` and `layers[1].weight` both become `("weight",)`) and producing key collisions. `jtu.keystr` is JAX's canonical string form for any path — distinct paths render to distinct strings, so the dispatch map stays unambiguous regardless of how the model mixes attribute, dict, and index access.

## Test the substrate

Six canonical patterns the substrate must support without forking.

```python
# 1. One INR per weight — params is the dict of INR modules themselves
def f(path, shape, dtype, params):
    inr = params[path]
    coords = ondes.nyquist_grid(shape)
    return jax.vmap(inr)(coords).reshape(shape)

inrs = {path: ondes.SIREN(...) for path in renderable_paths(P)}
rendered = loom.render(P, f, inrs)
```

```python
# 2. Shared functa body with per-leaf FiLM
def f(path, shape, dtype, params):
    body, films = params
    coords = ondes.nyquist_grid(shape)
    return jax.vmap(lambda c: body(c, film=films[path]))(coords).reshape(shape)

body = ondes.SIREN(...)
films = {path: jnp.zeros((d_mod,)) for path in renderable_paths(P)}
rendered = loom.render(P, f, (body, films))
```

```python
# 3. Hypernet over a target distribution (B targets, vmap over params)
def f(path, shape, dtype, params):
    coords = ondes.nyquist_grid(shape)
    return jax.vmap(lambda c: body_apply(c, params))(coords).reshape(shape)

rendered_batch = jax.vmap(loom.render, in_axes=(None, None, 0))(P, f, params_batch)
```

```python
# 4. Hypernet with per-target conditioning context (scan over (target, ctx) pairs)
def f(path, shape, dtype, params):
    body, films, target_ctx = params
    coords = ondes.nyquist_grid(shape)
    return jax.vmap(lambda c: body(c, film=films[path], ctx=target_ctx))(coords).reshape(shape)

def step(carry, target_ctx_t):
    params_t = (body, films, target_ctx_t)
    rendered_t = loom.render(P, f, params_t)
    return carry, compute_loss(rendered_t)

_, losses = jax.lax.scan(step, init, contexts_per_target)
```

```python
# 5. Inner-loop adaptation (iMAML / SAM / Reptile inner step)
def task_loss_of(p):
    return task_loss(loom.render(P, f, p))

def inner_step(p, _):
    return p - lr * jax.grad(task_loss_of)(p), None

adapted, _ = jax.lax.scan(inner_step, params, jnp.arange(K))
```

```python
# 6. Heterogeneous f per leaf-group (different INR for conv vs FC, no parallel structure)
def f(path, shape, dtype, params):
    conv_params, fc_params = params
    coords = ondes.nyquist_grid(shape)
    if "conv" in str(path):
        return jax.vmap(lambda c: conv_inr_apply(c, conv_params))(coords).reshape(shape)
    else:
        return jax.vmap(lambda c: fc_inr_apply(c, fc_params))(coords).reshape(shape)

rendered = loom.render(P, f, (conv_params, fc_params))
```

If any of these six needs a special primitive added to loom, the substrate has failed. If the user has to subclass anything, the substrate has failed. If the user has to attach a parallel meta-tree, the substrate has failed. If the user has to know what a `KeyPath` is to write the canonical case, the docs have failed (but the substrate has not).

## What this rules out from v1 (recap)

For the record, the following v1 concepts do *not* survive:

- `Condition` value type
- `Target` protocol with weight-filter and per-axis coord semantics
- `TaskCfg` / `TaskData`
- `RunResult` and `train_multi_seed`
- Optimiser groups and `clip_each_leaf`
- Pluggable diagnostics framework
- Plotting helpers
- `cartesian_axes` and generated condition registries
- Polar orthogonalisation as a loom primitive
- FiLM modulation as a loom primitive
- `coord_grid` kwarg or default builder (now `ondes` or user code)
- `should_render` predicate (now `eqx.partition`)
- `virtualise` primitive (structurally impossible for JIT / scan / grad; eager-Python lazy violates Guarantee 1)

These were trying to be a *renderer studio*. This document describes a *substrate*.
