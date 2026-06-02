# loom — Philosophy

*Draft v3, after six-voice review of v2 (data-scientist, ml-engineer, staff-architect, advocate, contrarian, pragmatist).
Resolves the three live disagreements from v2's open questions; folds in the team's structural additions.*

## The question

What is loom for?

## The verdict

**loom is a substrate for re-parameterising JAX pytrees by the output of arbitrary functions, in a way that is scannable and optimisable.**

That's it. Not a renderer for neural-network weights. Not a hypernet framework. Not a training studio. A *substrate*. The substrate is function-agnostic — INRs from `ondes` are the canonical instantiation, but loom does not bake them in. Anything that maps `path → array-of-leaf-shape` is a valid renderer.

Everything else — target architectures, task configs, training loops, diagnostics, plotting, sweep registries, coord-grid conventions, distribution heads — lives downstream. loom owns the mechanism, not any specific instantiation of it.

The library is thin by design. The product is the **contract** (§"Contract guarantees"), not the lines of code.

## Six principles

Each principle is followed by *what it rules out* — the negative space matters as much as the positive claim.

### 1. Simple over expressive

*Operationalised form (from staff-architect):* **every named concept must appear in the signature of a primitive a user calls.** That kills `Slot`, `LeafConditioning`, `Condition`, `Renderer`, `Target` reflexively — none of them appear in `render(P, f, params)`.

**Rules out**: value-type proliferation, registry abstractions, configuration objects, anything that grows the conceptual surface without earning it.

### 2. Mathematically transparent

*Falsifiable form (sharpened from v2 via staff-architect):* **if the maths says `W = f(meta)` for some leaf-identifying meta, the code reads `f(meta)` with nothing else in the signature.** Any deviation must be named in the docs as an explicit cost-paid.

The current substrate signature is `f(path, shape, dtype) → array`. `path` and `shape, dtype` are not in the maths — they are loom's contribution to make the call dispatchable. This is named cost: loom hands `f` the *identity* and *shape* of the leaf it's rendering, and gets back a tensor. The function may or may not use any of them.

```python
# What the maths says — the user thinks in these terms:
# f_path: () -> tensor of shape leaf.shape
# loom evaluates f_path() for each renderable leaf

# What the code reads:
loom.render(P, lambda path, shape, dtype: f_for(path).materialise(shape), params)
```

**Rules out**: ergonomic-but-impure wrappers, helper layers that hide the mathematical structure for brevity, anything where the user can't tell which lines are doing the maths.

### 3. Minimally invasive on the user's pytree

The user has a target pytree `P` (e.g. a CNN's weight tree). loom re-parameterises it without forcing the user to restructure `P`, attach parallel metadata trees, or wrap leaves in marker types. The substrate works *on* the user's pytree as-is. Identity is the native JAX pytree path (`jax.tree_util.KeyPath`); leaf shape and dtype come from `hasattr(leaf, 'shape')` duck-typing.

**Rules out**: parallel meta-trees, custom leaf-wrappers, `loom.Leaf(...)` marker types, `@loom.renderable` decorators, anything that asks the user to mutate their model definition. (The recipe in §"Refactor-safe dispatch" provides an escape hatch the user owns.)

### 4. Scan- and optimise-native

*Operationalised form:* `render`'s output is a pytree where `jax.tree_util.tree_structure(rendered) == jax.tree_util.tree_structure(P)` (when nothing is filtered) and all rendered leaves are `jax.Array` instances. Gradients flow `loss → P_rendered → params` via standard autodiff. `jax.vmap(render, in_axes=(None, None, 0))` over batched `params` works. `jax.lax.scan` with `render` in the body works.

*Static-target assumption (ml-engineer):* `render` traces `P` once per JIT cache key. Calling `render` under `vmap`/`scan` requires `P`'s structure to be identical across the mapped axis — only `params` varies. Heterogeneous target structures need separate `render` calls. *Cross-axis target heterogeneity is not in scope.*

**Rules out**: custom gradient-flow APIs, "loom-aware" optimisers, anything that requires `jax.lax.scan(loom.scan_step, ...)` instead of plain `jax.lax.scan`.

### 5. Mechanism, not policy

loom does not interpret what the user's function *is*. It does not know whether `f` is an INR, a hypernetwork, a closed-form mapping, or a lookup table. It does not own coord-grid construction, distribution choices, conditioning conventions, or filtering policy. The user composes `ondes` (or anything else) into `f`; loom hands `f` the leaf's identity and shape, takes back the value, and threads gradients.

**Rules out**: `loom.heads`, `loom.distributions`, `loom.basis`, `loom.coord_grid`, `loom.nyquist_*`, anything that ships opinionated primitives for what `f` should look like. ondes ships those (where applicable). loom doesn't re-export them.

### 6. Composable, not configurable

*From staff-architect.* When something needs to vary — coord-grid choice, filtering, multi-target batching, memory checkpointing — the answer is *user composes loom with JAX primitives + their own `f`*, not *loom grows a kwarg or a flag*. See §"Composition recipes" for the canonical patterns.

**Rules out**: future kwarg-bloat. Every PR proposing a new `render(..., mode='X')` flag has to first show why composition with `jax.checkpoint`/`jax.vmap`/`jax.lax.scan`/`eqx.partition`/user-side dispatch can't express it.

## The primitive

loom v3 has **one primitive**:

```python
def render(
    P: PyTree,
    f: Callable[[KeyPath, Shape, DType], Array],
    params: PyTree,
) -> PyTree:
    """Materialise the pytree.

    For each leaf at `path` in `P` where `hasattr(leaf, 'shape')`:
        rendered_leaf = f(path, leaf.shape, leaf.dtype)
        # invariant: rendered_leaf.shape == leaf.shape

    For non-shape-bearing leaves: pass through unchanged.

    Returns a pytree with the same structure as P; renderable leaves
    replaced by f's output, others preserved.

    Gradient flows from rendered through f's closure into `params` via
    standard JAX autodiff.
    """
```

That is the entire public surface. Three positional args, no kwargs. The positional ABI `(P, f, params)` is the stable API — future additions, if any, only come after `params`.

## Composition recipes

The patterns that v2 considered as second primitives are JAX-primitive compositions in v3:

```python
# Memory-efficient backward (v2's open Q6 — now downstream)
loss = jax.checkpoint(lambda p: model(loom.render(P, f, p), x, y))(params)

# Batched targets / hypernet over a distribution (v2 test pattern 3)
rendered_batch = jax.vmap(loom.render, in_axes=(None, None, 0))(P, f, params_batch)

# Inner-loop adaptation (e.g. iMAML / SAM inner steps)
def inner_step(p, _):
    rendered = loom.render(P, f, p)
    return p - lr * jax.grad(task_loss)(rendered), None
adapted, _ = jax.lax.scan(inner_step, params, jnp.arange(K))

# Selective rendering (v2's `should_render` kwarg — now a pre-split)
renderable, passthrough = eqx.partition(target, filter_spec)
rendered = loom.render(renderable, f, params)
final = eqx.combine(rendered, passthrough)

# Lazy materialisation for edge inference (v2's `virtualise` — now downstream)
# Inside eager Python, not under jax.jit:
def leaf_thunk(path, shape, dtype):
    return lambda: f(path, shape, dtype)
P_lazy = loom.render(P, leaf_thunk, params)  # leaves are now 0-arg callables
# downstream consumer calls leaf() to materialise
```

If a recipe takes more than five lines to express, *that's* the gap to consider closing — not by adding to loom, but by adding to whichever sibling library is responsible (ondes for INR composition, fws for diagnostics, downstream for studio behaviour).

## Contract guarantees

The product loom sells is not the implementation — it's these guarantees:

1. **Structure preservation.** `tree_structure(render(P, f, params))` equals `tree_structure(P)` (on renderable leaves) or returns the same leaf unchanged (on non-shape-bearing leaves).
2. **Shape correctness.** For every renderable leaf, the rendered output has `.shape == leaf.shape`. Loom raises `loom.ShapeMismatch` with a path-pointing error message when `f` violates this.
3. **dtype rule.** Rendered output dtype is `f`'s return dtype. Loom does not cast. The user's `f` is responsible for matching the model's expected dtype.
4. **vmap-axis convention.** None — loom does not impose a coord-grid axis convention. `f` is called once per leaf; how `f` constructs its inputs internally is `f`'s business.
5. **Gradient flow.** Standard JAX autodiff applies. `params` flows into `f`'s closure; gradients flow back via the same closure. Loom does not intercept gradients.
6. **JIT-stability.** Under `jax.jit`, the per-leaf Python loop in `render` is traced once per cache key (set by `P`'s structure). The static-target assumption applies.

These six are testable; the test suite verifies them on a synthetic minimal pytree before any release.

## What loom owns

- `render` (the primitive above)
- The contract guarantees (above)
- The `KeyPath` convention as the leaf-identity type
- `loom.ShapeMismatch` exception type

## What loom does NOT own

- Coord-grid construction or conventions (Nyquist, linspace, dyadic, anything else) — `ondes` or user code
- Filtering / partitioning of which leaves to render — `eqx.partition` and user composition
- Lazy materialisation under `jit`/`scan`/`grad` — structurally impossible (JAX traces materialise at trace time); eager-Python lazy is a 3-line user recipe
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

Things loom *deliberately* does not do, to prevent feature drift:

- **Cross-leaf coupling.** Each leaf is rendered as a pure function of `(path, shape, dtype, params)`. If your renderer needs leaf B's rendered output to inform leaf A's render, that coupling lives in `params` or is computed downstream after `render` returns. Loom does not ship `render_sequential`, `render_with_dependencies`, or any DAG-aware variant.
- **Loss computation.** `render` returns a pytree. What you do with it (forward pass, loss, backprop) is downstream. No `render_and_loss`, no `render_and_grad`.
- **Lazy materialisation inside JIT/scan/grad.** Structurally impossible per JAX trace semantics. Eager-Python lazy is a user recipe, not a primitive.
- **Heterogeneous target structures across `vmap`/`scan` axes.** The static-target assumption is load-bearing.
- **Distribution / variational machinery.** `f` may stochastically sample (closing over a key); loom does not interpret stochasticity, KL terms, or ELBOs.

## Refactor-safe dispatch (recipe, not API)

The native identity is `KeyPath`. Path is fragile under field-renames. The substrate accepts this cost in exchange for Principle 3 (no wrapper types). Users who need refactor-safe dispatch own a small adapter:

```python
# User-side: stable tag map, owned by the user, lives next to their model
tag_of_path = {
    ("conv1", "weight"): "conv_first",
    ("conv2", "weight"): "conv_second",
    ("head",  "weight"): "head",
}

# User's f dispatches on tag, not path
def f(path, shape, dtype):
    tag = tag_of_path[tuple(p.name for p in path if hasattr(p, 'name'))]
    return inrs[tag].materialise(shape, dtype, params=inr_params[tag])

rendered = loom.render(P, f, inr_params)
```

If the user renames `conv1` to `conv_a` in their model, they update one line in `tag_of_path`. Their `inrs` dict, dispatch logic, and call site remain untouched. This is a *5-line user pattern*, not a loom primitive — it preserves Principle 3 and gives the escape hatch to anyone who needs it.

## Test the substrate

Six canonical patterns the substrate must support without forking. Each is a working JAX program against the proposed signature.

```python
# 1. One INR per weight
inrs = {path: ondes.SIREN(...) for path in renderable_paths(P)}
def f(path, shape, dtype):
    coords = ondes.nyquist_grid(shape)
    return jax.vmap(inrs[path])(coords).reshape(shape)
rendered = loom.render(P, f, inr_params)
```

```python
# 2. Shared functa with per-leaf FiLM
body = ondes.SIREN(...)
films = {path: jnp.zeros((d_mod,)) for path in renderable_paths(P)}
def f(path, shape, dtype):
    coords = ondes.nyquist_grid(shape)
    return jax.vmap(lambda c: body(c, film=films[path]))(coords).reshape(shape)
rendered = loom.render(P, f, (body_params, films))
```

```python
# 3. Hypernet over a target distribution (B targets, vmap over params)
def f(path, shape, dtype):
    coords = ondes.nyquist_grid(shape)
    return jax.vmap(lambda c: body(c, params=body_params))(coords).reshape(shape)
rendered_batch = jax.vmap(loom.render, in_axes=(None, None, 0))(P, f, params_batch)
```

```python
# 4. Hypernet with per-target conditioning context (scan over (target, ctx))
def step(carry, target_ctx_t):
    def f_ctx(path, shape, dtype):
        coords = ondes.nyquist_grid(shape)
        return jax.vmap(lambda c: body(c, film=films[path], ctx=target_ctx_t))(coords).reshape(shape)
    rendered_t = loom.render(P, f_ctx, (body_params, films))
    return carry, compute_loss(rendered_t)
_, losses = jax.lax.scan(step, init, contexts_per_target)
```

```python
# 5. Inner-loop adaptation (iMAML / SAM / Reptile inner step)
def inner_step(p, _):
    rendered = loom.render(P, f, p)
    return p - lr * jax.grad(task_loss)(rendered), None
adapted, _ = jax.lax.scan(inner_step, params, jnp.arange(K))
```

```python
# 6. Heterogeneous f per leaf-group (different INR for conv vs FC, no parallel structure)
def f(path, shape, dtype):
    coords = ondes.nyquist_grid(shape)
    if "conv" in str(path):
        return jax.vmap(lambda c: conv_inr(c, params=conv_params))(coords).reshape(shape)
    else:
        return jax.vmap(lambda c: fc_inr(c, params=fc_params))(coords).reshape(shape)
rendered = loom.render(P, f, (conv_params, fc_params))
```

If any of these six needs a special primitive added to loom, the substrate has failed. If the user has to subclass anything, the substrate has failed. If the user has to attach a parallel meta-tree, the substrate has failed. If the user has to know what a `KeyPath` is to write the canonical case, the docs have failed (but the substrate has not).

## What this rules out from v1 (recap)

For the record, the following v1 concepts do *not* survive v3:

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
- `virtualise` primitive (now user recipe for eager-Python, structurally impossible for JIT/scan/grad)

These were trying to be a *renderer studio*. v3 is a *substrate*.

## Remaining open question

Only one survives the six-voice review:

**Q1. Does the rendered output need to enforce dtype-equality with `P`'s leaf dtypes?** Currently the contract says "rendered dtype is `f`'s return dtype; loom does not cast". This is mechanism-not-policy clean but means downstream consumers must explicitly cast if their model expects a specific dtype. Alternative: loom casts to `P`'s leaf dtype after `f` returns. Trade-off: ergonomic but opinionated.

My lean: leave it as-is (no cast). The user composing `f.astype(...)` or `jax.tree_util.tree_map(lambda l, r: r.astype(l.dtype), P, rendered)` is one line. Worth confirming with Ammar.
