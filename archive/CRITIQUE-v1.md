# CRITIQUE — `loom` decomposition

Contrarian read on the premise of the package, written before reading
the architect's `DESIGN.md`. The goal is independent failure-mode
identification, not adversarial criticism: where I push back, I want
the synthesis pass to weigh both framings.

**Scope (settled, per team-lead clarification 2026-05-15)**: `loom` is a
reusable library peer to `samgria` / `rltrain` / `xptrack`. It owns the
**machinery**, not the **instantiations**.

| in `loom` | not in `loom` |
|---|---|
| `Basis` ABC + SIREN/HSIREN/WIRE | `FCHeavyCNN`, `IrisMLP`, `ResidualConvNet` |
| Renderer machinery (virtualise, render, polar_orthogonalise, FiLM, slots, leaf conditioning) | `load_digits_task`, `load_iris_task`, `load_cifar10_task` |
| `Encoding` ABC + Gaussian / Dyadic / Learned | The pre-populated `CONDITIONS = {...}` registry content |
| `Condition` value type, `Target` / `TaskCfg` **protocols** | Runner scripts (`run_basin_study.py` et al.) |
| Training primitives (`train_multi_seed`, `RunResult`, optimiser groups, `clip_each_leaf`) | |
| Diagnostics primitives, plotting helpers | |

The non-library bits move to `loom/examples/` or downstream
`experiment/`. Same principle as rltrain's no-env-deps rule and
samgria's no-training-loop rule.

This **sharpens** the critique. The downstream consumer is now
load-bearing: they instantiate `Target` and `TaskCfg` from outside the
library, they fill the `CONDITIONS` registry themselves, they write
the capacity sweep loop themselves. Three concrete user stories drive
the rest of this document:

- **Ammar / future FWS collaborators** running the capacity sweep next
  week. They want `for hidden in [12, 16, 24, 48, 96]: ...` to be
  ergonomic over the four real axes (basis × encoding × ortho ×
  capacity).
- **A hypothetical second consumer** writing a transformer-weight
  renderer. They define their own `Target` (with attention heads),
  their own `Task`, and a new `Basis` if needed.
- **The FWS Paper-2 transfer-learning angle** (per memory): cross-task
  σ comparison, which means iterating over (target, task) pairs the
  library does not enumerate.

If `loom` makes any of those three stories require forking or
monkey-patching, it has failed its scope claim.

## 1. Extensibility — the highest-leverage critique

The library's extension surface is the only thing downstream consumers
will touch. The monolith encodes extension by editing the module —
that won't survive becoming a library. Walk through each axis and look
at the seams.

### 1a. `Basis` — adding a new activation family

The monolith's basis dispatch is a tagged union by string:

```python
# experiment.py:91–98
if self.kind == "siren":  return jnp.sin(self.omega * pre)
if self.kind == "hsiren": return jnp.sin(self.omega * jnp.sinh(pre))
if self.kind == "wire":   return jnp.cos(...) * jnp.exp(...)
raise ValueError(f"unknown basis kind {self.kind!r}")
```

`BASIS_KINDS = ("siren", "hsiren", "wire")` (line 45) is a module-level
tuple. The `s` parameter is stored on every layer ("for pytree
uniformity") even though only WIRE uses it. Adding a "gabor-asymmetric"
basis requires editing `BasisLayer.__call__`, editing `BASIS_KINDS`,
threading any new per-basis parameter through `BasisLayer.__init__`,
`BasisBody.__init__`, `make_shared`, and `shared_condition` — five
files for a one-line mathematical change.

In a library, that pattern means **forking**. The downstream consumer
cannot add a basis without editing `loom`. I'd push back hard on this
surviving the refactor. The honest API is a protocol:

```python
class Basis(eqx.Module, Protocol):
    """A pointwise activation z -> a(z), with learnable scalar params."""
    def __call__(self, pre: Array) -> Array: ...
```

`SIREN`, `HSIREN`, `WIRE` become concrete `Basis` modules in
`loom.basis`. `BasisLayer` becomes generic over a `Basis` factory. The
"pytree uniformity" reason for storing an unused `s` on every layer
evaporates because each basis owns its own fields.

Second-order benefit: the **slow-param labelling** (line 930,
`SLOW_NAMES = {"omega", "s", "sigma_learnable"}`) is currently a
hardcoded set of attribute names. If a new basis introduces a new
slow-natured parameter, it has to be added to that string set in a
separate file. With a `Basis` protocol, each basis can declare its own
slow params (a class attribute, an `eqx.field` annotation, or a
`slow_param_names() -> set[str]` class method). Extension stays local.

### 1b. `Encoding` — adding a new input-side preprocessing

`LeafConditioning` (lines 142–229) dispatches on `encoding_kind` as a
string. The `fourier_B` array, `sigma_learnable` scalar, and
`num_bands` int are stored on every instance regardless of which
encoding is active (lines 168–170 — "kept for pytree uniformity"). A
new encoding (say, "wavelet" with its own scale/translation params)
must:

1. Edit the `if/elif/else` in `__init__`.
2. Edit the `if/elif/else` in `project`.
3. Add dummy-init branches for the existing two encodings to keep
   pytree uniformity.
4. Edit the `Encoding` dataclass to include any new params.
5. Edit the constructor helpers (`gaussian_fixed`, `dyadic`, etc.).

This is where I most expect to disagree with the architect: the easy
move is to lift `Encoding` as a dataclass with `kind: str` and keep
the dispatch. That's *configuration*, not *extension*. A downstream
consumer who needs a 4th encoding then has to fork.

The honest pattern is the same as §1a: `Encoding` is a protocol with a
`project(coord) -> Array` method, and `LeafConditioning` becomes a
composition: `Encoding . HeadProjection . FiLM`. Each `Encoding`
implementation owns its own state. No more dummy `fourier_B` arrays on
`Dyadic` instances.

### 1c. `Target` protocol — the load-bearing test

Per the narrower-scope clarification: `Target` and `TaskCfg` are
**protocols**, not concrete impls. Concrete targets (`FCHeavyCNN`,
`IrisMLP`, etc.) live in `loom/examples/` or downstream. This makes
the `Target` protocol the **only** extension point for new
architectures — which means it has to actually work for architectures
the monolith was not written against. The audit question
team-lead asked: does the current `_is_weight` / `_render_via_basis` /
`_render_leaf` machinery work for a transformer's
`nn.MultiheadAttention`? Walk it.

A standard `nn.MultiheadAttention` has at least three weight tensors
of concern:

- **Q/K/V projections** — rank-2 (model_dim × head_dim × n_heads,
  typically reshaped to `(n_heads * head_dim, model_dim)`). These look
  like ordinary `Linear` weights to `_is_weight`. **Renders.**
- **Output projection** — rank-2. **Renders.**
- **Relative-position bias table** (in many transformer variants) —
  rank-2 `(num_buckets, num_heads)`. Looks rank-2 to `_is_weight` but
  carries position-indexed structure that the coord-smoothness prior
  is *actively wrong* for: adjacent buckets correspond to adjacent
  relative positions in *one* axis, but the second axis (heads) is
  permutation-symmetric. **Renders silently and badly.**
- **LayerNorm gamma/beta** — rank-1 learnable per-channel scale/shift.
  `_is_weight` requires `non_singleton >= 2` (line 322), so these are
  excluded. **Stays as direct params.** Is that correct? In some
  transformer renderer schemes you'd *want* to render these (the
  per-channel scale is exactly the kind of low-frequency structure a
  renderer should capture). Currently no way to opt them in.

So the audit yields three concrete failures of the current machinery
against the transformer case:

1. **No per-target override for `_is_weight`.** The decision "what
   tensors are renderable?" is hardcoded at module level. A downstream
   consumer who wants rank-1 gammas in or rank-2 position-biases out
   has to monkey-patch.
2. **No per-leaf shape semantics.** The coord-smoothness prior assumes
   every tensor axis is a continuous index. For position-bias tables,
   axis 0 is continuous and axis 1 is categorical/permutation-
   symmetric. The renderer has no way to express "render along this
   axis, average along that one" — the `_normalized_grid` function
   (lines 306–309) blindly produces a Cartesian grid on every axis.
3. **No coord-frame override.** The grid is always
   `linspace(-1, 1, k)` per axis. For axes where the natural indexing
   is logarithmic (a relative-position bucket table with log-spaced
   buckets, for instance) or one-sided, the consumer can't override
   the coord mapping.

The lift here is concrete:

```python
class Target(Protocol):
    def weight_filter(self) -> Callable[[Any], bool]: ...
    def slot_axes(self, weight: Array) -> tuple[Axis, ...]: ...
    # Axis specifies: continuous | categorical | logarithmic, range, ...
```

Defaults match today's behaviour (rank >= 2 -> continuous on every
axis, `linspace(-1, 1, k)` grid). Consumers override.

This is the audit-test the team-lead asked for, and the answer is:
**as it stands, `loom` cannot render a transformer without
monkey-patching**. That's a scope-claim failure for a library
positioned as the renderer for FWS-adjacent architectures.

### 1d. `TaskCfg` protocol — fewer pitfalls but one trap

`TaskCfg` is already close to right: a dataclass with `name`,
`template_fn`, `loader`, `num_steps`, `batch_size`, `lr`. The protocol
version is straightforward.

One trap: `loader()` returns `TaskData` with hardcoded fields
`xs_train`, `ys_train`, `xs_test`, `ys_test`, `name` (line 458).
That's fine for classification-on-fixed-data, but RL policy-net
rendering (per the scope claim) has no `(xs, ys)` — it has a `gym.Env`
and trajectory batches. The protocol either needs to be generic over
the data shape, or the library needs a `TaskData` protocol that the
classifier-style `TaskData` happens to implement.

I'd push back on baking the classification data shape into the
library. Either:

- Make `TaskData` a protocol (`train_batch() -> Any`,
  `eval_batch() -> Any`), or
- Drop `TaskData` from `loom` entirely and let `train_multi_seed`
  take a `batch_fn: Callable[[Key], (params_batch, loss_args)]` that
  the consumer constructs from whatever data they have.

The second is cleaner — `loom` shouldn't know anything about
"train/test split" as a concept. That's a classification-experiment
convention, not a renderer-machinery one.

### 1e. `Diagnostic` — pluggable, not hardcoded

`_group_grad_norms` (lines 1022–1043) and `_cross_seed_cosine_scalar`
(lines 994–1019) are baked into the inner training scan (lines 1167–
1172). A downstream consumer who wants to record, say, per-layer
rendered weight spectral norm over training has no extension path —
they fork `train_multi_seed`.

Push back: diagnostics is a `tuple[Diagnostic, ...]` parameter on
`train_multi_seed`. Each `Diagnostic` is
`(params, batch, *) -> jax.Array` and gets stacked into `RunResult`.
The current two diagnostics become defaults. New consumers add their
own.

### 1f. Summary of extensibility critique

The monolith is shaped like a **closed factorial study**: every axis
is a string-tagged enum, every registry is a module-level constant,
every diagnostic is hardcoded. For a one-paper experiment that's
fine. For a reusable library positioned as peer to
`samgria`/`rltrain`/`xptrack`, it's an extension-by-fork pattern.

The lift is concrete: replace each `if kind == ...` chain with a
protocol; replace each module-level `*S = [...]` list with a function
parameter; replace `_is_weight` with a target-supplied filter; replace
the hardcoded grid with target-supplied axis semantics.

The transformer-audit failure in §1c is the load-bearing example —
that's the test the library has to pass to earn its scope claim.

## 2. The unstated assumptions

Three framings (orthogonal to §1) I expect the architect to default
to, each of which I would push back on.

### 2a. "The existing module boundaries are correct."

The monolith has banner-comment section dividers (`=== BASES ===`,
`=== TASKS ===`, `=== CONDITIONS ===`, etc.) that read like a
ready-made module list. The temptation is to map each banner to a
file.

I would push back on at least two boundaries:

- **`BasisLayer` / `BasisBody` belong together, but `LeafConditioning`
  is *not* a "basis"** — it's the input head + FiLM modulation, and
  it owns three independent encoding modes. The "basis" banner
  conflates the inner-MLP activation family and the input encoding.
  They are independently varied in the `Condition` axes
  (lines 793–832). If the package adopts banner-as-module, encoding
  ends up in the wrong file and the three-axis story falls apart at
  the API level. `basis` and `encoding` are sibling modules.
- **`virtualize_standard` vs `virtualize_per_leaf`** look like two
  variants in the monolith (lines 401–427), but their `LeafSlot` /
  `PerLeafSlot` types are almost the same — `PerLeafSlot` is
  `LeafSlot + body`. If the package keeps them parallel, every
  consumer type-dispatches on the slot (which `_render_leaf` already
  does, lines 433–444). A single `Slot` with an optional per-leaf
  body collapses the dispatch — and would also make the body-capacity
  question (§3 below) coherently expressible as "what body does this
  slot use?" rather than "is this a shared or per-leaf rendering?".

### 2b. "It needs a flat `__init__.py` with everything re-exported."

The monolith's consumers do `import experiment as E` and reach in by
name. About 20–30 symbols. The default refactor preserves that flat
namespace via `loom.__init__` re-exports.

I'd push back. Two failure modes:

- Re-export everything -> package looks flat but is structurally
  nested. Every refactor inside the package risks a downstream import
  break.
- Curate a smaller public surface -> downstream scripts break
  immediately, because the monolith pattern is to reach into
  `E.CONDITIONS` and `E.EVAL_EVERY` directly.

The honest move is **callers import from submodules**:
`from loom.basis import SIREN`, `from loom.render import render`. The
module structure becomes load-bearing — inspectable — instead of
hidden behind a flat namespace pretending to be stable.

Override if `samgria` / `rltrain` / `xptrack` already use flat
re-exports — peer-library consistency wins over this rule.

### 2c. "The three-axis Condition factoring is canonical."

The three-axis Condition (basis × encoding × ortho) landed
**yesterday** (2026-05-15, CHANGELOG "Parameter-free polar
orthogonalisation + three-axis condition API"). It is the most recent
abstraction in the codebase, on a 10-iteration design history.

§11 of the findings doc already names **body capacity** as a fourth
axis the abstraction doesn't own. See §3. Building the three-axis
`Condition` into `loom`'s public surface as the canonical sweep
primitive means the library ships with an abstraction known to be one
axis short.

## 3. Body capacity must be a first-class Condition axis

§11 of the findings doc admits the capacity gap explicitly: the body
MLP has been `SIREN_HIDDEN=24, SIREN_LAYERS=2` throughout, every
empirical claim is conditional on that one point in capacity space,
and the Phase 1 capacity scan is more central to FWS than the §8 next
experiments.

Per the narrower-scope clarification, the capacity sweep is no longer
a question of "does `loom` configure this?" — `loom` provides the
machinery, the downstream consumer writes the sweep loop. So the
question becomes **API ergonomics**: is the consumer's capacity-sweep
loop natural to write, or does it require workarounds?

The ergonomic test is concrete. Imagine the experiment author writing:

```python
results = {}
for hidden in [12, 16, 24, 48, 96]:
    for basis in [SIREN(), HSIREN(), WIRE()]:
        for encoding in [Identity(), Gaussian(sigma=PI), Dyadic(L=4)]:
            for ortho in [False, True]:
                cond = Condition(basis=basis, encoding=encoding,
                                 ortho=ortho, body=Body(hidden=hidden))
                results[cond.id] = train_multi_seed(cond, ...)
```

That four-nested-loop pattern needs to be **natural**, not
boilerplate-heavy. Three concrete API demands fall out:

- **`Condition` must accept body capacity on equal footing with
  basis/encoding/ortho.** Not as a hidden kwarg with a default — as a
  named, typed axis. The monolith's `body_hidden` / `body_layers` are
  module-level constants (`SIREN_HIDDEN`, `SIREN_LAYERS` at lines
  702–703); lifting them into a `Body` value type is the natural
  shape.
- **`Condition` needs a stable `id` / hashable form** so the
  consumer's results dict has a meaningful key. Currently the
  identification is by string-key into a module-level `CONDITIONS`
  dict — that breaks the moment the sweep is Cartesian.
- **`loom.sweep.cartesian_axes(*axes)` helper.** Not strictly required
  — the consumer can write the nested loops themselves — but the
  ergonomics of
  `for cond in cartesian_axes(bases, encodings, orthos, bodies): ...`
  is significantly better than four levels of nesting, and once you
  have it the same helper covers the
  `(target × basis × encoding × ortho × capacity)` five-axis case for
  cross-task transfer (Paper-2).

The push-back here: if the architect exposes capacity as a kwarg on
`Condition` rather than as a named axis with a value type, the
consumer's sweep loop becomes:

```python
for hidden in [12, 16, ...]:
    cond = shared_condition(basis=..., encoding=..., ortho=...,
                            body_hidden=hidden)
```

That works for capacity-as-kwarg but doesn't generalise. When the next
paper needs a sweep over `body_layers` *and* `body_hidden`, the kwarg
list grows. When the consumer wants to print "which axis varied for
this row?", there's no way to ask the `Condition` what it is — the
body is a hidden internal detail. The kwarg pattern is the
one-axis-short problem in micro.

The target architecture is implicitly a sixth axis. `for target in
[DigitsCNN(), CifarCNN()]:` should be the same pattern. That's what
makes `loom.sweep.cartesian_axes` (or whatever the architect calls it)
the natural-convergence point — it serves both the capacity sweep
*and* the transformer-second-consumer story (where their sweep is
over `(my_target, my_basis_variant, ...)`).

## 4. Migration cost vs design quality tradeoff

Per the narrower-scope clarification, the migration is smaller than I
had previously implied. With concrete targets, loaders, and runners
moving to `examples/` (or staying in `experiment/`), the runner
scripts become thin: `import loom` + import the local example targets
+ import the local loaders + drive the sweep.

The remaining cost:

- Six runner scripts (`run_basin_study.py`, `run_followup_study.py`,
  `run_sigma_probe.py`, `run_cifar_validate.py`, `run_probe.py`,
  `build_validation_notebook.py`) — rewrite their imports and replace
  `E.CONDITIONS[ck]` reach-ins with explicit construction. Mechanical.
- The `CONDITIONS = {...}` dict literal (lines 868–910, 40 lines) —
  delete from `loom`, recreate in `experiment/` or in each runner that
  needs a canned sweep.
- Re-run experiments to confirm parity (per `test-your-claims.md` —
  "it runs without error" is not proof of correctness). The seeded
  determinism (memory: "master_seed=42 plus deterministic JIT means
  replications cost nothing") makes this cheap.

The contrarian timing question I want on the table:

1. **Decompose now**. Pay the migration cost up front. §3
   (capacity-as-axis, generated registry) becomes load-bearing so the
   library doesn't ship with a known-one-axis-short abstraction.
2. **Run the capacity sweep in the monolith first**. Add
   `SIREN_HIDDEN` and `SIREN_LAYERS` as kwargs on
   `shared_condition` (~5 lines). Run the sweep. Then decompose
   `loom` with four-axis empirical data in hand.

The case for option 2: doing the decomposition *after* the capacity
sweep means the `Condition` abstraction is designed against pressure
that has actually been applied, not against pressure that's
hypothetical. The user's preference is "clean mathematical
abstractions" — that preference is easier to honour when the empirical
pressure has hit the abstraction at least once. A week of delay is
cheap; a locked-in three-axis Condition that needs reshaping in month
two is expensive.

The case for option 1: the runner scripts already use the
layer-violating `import experiment as E; E.CONDITIONS = ...` pattern
(per memory `project_loom.md`), so any further work in the monolith
inherits the same fragility. The cleanest path is to do the
decomposition first, then run the sweep against the new library.

I lean option 2, but it's a close call. The deciding question is:
**how confident is the architect that the three-axis Condition will
survive contact with the capacity sweep?** If "very", option 1 is
fine. If "let me think about that", option 2 wins.

## 5. What I would do differently — module sketch

A counter-sketch at the module-name level, narrower-scope-aware.
Callers import from submodules directly; `__init__.py` is empty or
declares only `__version__`. `loom/examples/` holds concrete targets
and tasks; experiment-specific runners live in `experiment/`.

| module | what it owns | stability |
|---|---|---|
| `loom.basis` | `Basis` protocol, `SIREN`, `HSIREN`, `WIRE`, `siren_init` | Stable. Each basis is a concrete class implementing the protocol. |
| `loom.encoding` | `Encoding` protocol, `Identity`, `Gaussian`, `Dyadic`, `LeafConditioning`, `nyquist_sigma` | Stable. Each encoding owns its own state. `LeafConditioning` composes `Encoding` + head projection + FiLM. |
| `loom.ortho` | `polar_orthogonalise` | Stable. One function. |
| `loom.slot` | `Slot` (unified — see 2a), `Axis` value type, `siren_in_dim_for`, `target_init_scale`, `default_weight_filter` | Virtualisation primitive. Per-axis semantics owned by `Axis` (continuous / categorical / log-spaced). |
| `loom.render` | `virtualize`, `render`, `render_leaf` | The functional render core. Takes a slot tree + optional shared body. |
| `loom.target` | `Target` protocol, `Body` value type (hidden, layers, basis-factory) | Pure protocol module. Concrete targets are downstream. |
| `loom.task` | `TaskCfg` protocol, `TaskData` protocol (or none — see 1d), `Diagnostic` protocol | Pure protocol module. Concrete tasks/data are downstream. |
| `loom.sweep` | `Condition` value type (basis × encoding × ortho × body × ...), `cartesian_axes`, `Axis` (sweep-level: a named list of values), `RunResult` | The sweep primitive. Generated registry over named axes. `RunResult` carries the full `Condition` back. |
| `loom.train` | `train_multi_seed`, `make_optimizer`, `clip_each_leaf`, slow-param machinery driven by `Basis.slow_param_names()` | Training harness. Takes a `Condition`, a `Target`, a `TaskCfg`, and a `tuple[Diagnostic, ...]`. |
| `loom.diag` | `group_grad_norms`, `cross_seed_cosine`, `count_params` | Default `Diagnostic` implementations. |
| `loom.plot` | `plot_loss_curves`, `plot_test_acc_curves`, `plot_final_bars`, `plot_grad_dynamics`, `plot_cross_seed_cos`, `summarize` | Decomposed plotters. Each takes a `RunResult` collection + the condition subset to render. No `DYNAMICS_CONDITION_SUBSET` module-level constant. |

`loom/examples/` (or `experiment/`):

| location | content |
|---|---|
| `loom/examples/targets.py` | `FCHeavyCNN`, `IrisMLP`, `ResidualConvNet`, the `DigitsCNN` / `CifarCNN` / `DigitsDeepCNN` / `CifarDeepCNN` partials |
| `loom/examples/tasks.py` | `load_digits_task`, `load_iris_task`, `load_cifar10_task` |
| `experiment/run_basin_study.py` etc. | Stays as-is, but imports from `loom` + `loom.examples` (or local equivalents) instead of `experiment.py` |

Notable changes from the monolith:

- **`basis` and `encoding` are sibling modules**, not nested under one
  banner.
- **`Slot` is unified** (§2a). `_render_leaf` type dispatch goes away.
- **`Axis` semantics** are explicit per slot (continuous / categorical
  / log-spaced) so transformer position-bias tables can be rendered
  correctly (§1c).
- **`Body` is a value type** in `loom.target`, capturing
  hidden/layers/basis-factory. `Condition` carries a `Body` instance.
- **No module-level `TASKS` / `CONDITIONS` / `BASIS_KINDS`
  constants.** Callers compose explicitly.
- **`sweep` owns the registry pattern.** `train` doesn't import
  `sweep`; it just takes a `Condition` instance. The Cartesian builder
  is in `sweep` so consumers' loops are one-line.
- **`diag` is pluggable.** Consumers add their own `Diagnostic`
  implementations without touching `train`.

## 6. The single biggest risk — concrete failure mode at 3 months

If `loom` ships with the current monolith's extension pattern
preserved (tagged-union dispatch, hardcoded `_is_weight`, fixed
coord-grid, body capacity as a kwarg), here's the failure path.

**Month 1**: capacity sweep paper figure. Ammar writes the
four-nested sweep loop. The `Condition` doesn't carry `Body`
explicitly, so the result dict's keys are awkward —
`(basis_name, encoding_name, ortho, hidden, layers)` tuples manually
constructed at every callsite. `RunResult` doesn't know which axis
varied, so the plotting code grows its own axis-tracking. Workable
but boilerplate-heavy.

**Month 2**: FWS Paper-2 cross-task transfer (per memory). The sweep
now ranges over (target, task) pairs too. The `Condition` doesn't own
the target axis, so a parallel registry appears in `experiment/`.
There are now two ways to describe an experiment: one in `loom`
(basis/encoding/ortho) and one in `experiment/` (target/task). Result
serialisation has to handle both.

**Month 3**: hypothetical transformer-renderer consumer (the scope
claim's load-bearing user). They write their `Target` with attention
heads. `_is_weight` silently renders the position-bias table as if it
were a continuous-grid weight, producing garbage. They debug, find
the hardcoded predicate, and either (a) monkey-patch `loom._is_weight`
from their consumer code (gross, version-fragile), or (b) fork the
library. Either way the scope claim "peer to
samgria/rltrain/xptrack, used by FWS-adjacent projects" is broken by
month three.

The throughline: **every new study and every new consumer reveals the
API was shaped to the 2026-05-15 snapshot**. `loom` and the science
diverge. The user spends weekly debate-time on whether to patch the
library or write around it — the cost the library was meant to
eliminate.

The path that avoids this: §1 (protocol-based extension), §3
(capacity-as-axis with a `Body` value type and a Cartesian-product
builder), and option 2 in §4 (sweep first, decompose second).

---

## Summary for the synthesis pass

1. **Extensibility is the highest-leverage critique.** Replace each
   tagged-union dispatch with a protocol. `Basis`, `Encoding`,
   `Diagnostic`, weight-filter, axis semantics. The transformer-audit
   in §1c shows `loom` cannot render a transformer as-is — that's a
   scope-claim failure for a library positioned as peer to
   samgria/rltrain/xptrack.
2. **Body capacity must be a first-class axis** carried as a `Body`
   value type on `Condition`, not a kwarg. The library provides
   `cartesian_axes(*axes)` so the downstream consumer's sweep loop is
   one line per axis. Target architecture is implicitly a sixth axis;
   the same primitive handles it.
3. **Module boundaries**: `basis` and `encoding` are siblings, not
   nested. `LeafSlot` + `PerLeafSlot` collapse into one `Slot` with
   optional body. `Axis` semantics (continuous / categorical /
   log-spaced) are per-slot, not hardcoded.
4. **No flat `__init__.py` re-exports** at this maturity — unless
   peer libraries (`samgria`, `rltrain`, `xptrack`) already do flat
   re-exports, in which case consistency wins.
5. **Migration timing**: I lean toward running the capacity sweep in
   the monolith first (option 2 in §4), then decomposing with
   four-axis empirical pressure already applied. Close call; option 1
   is fine if the architect is confident the three-axis Condition
   survives the sweep.

These are framings to *weigh* against the architect's, not to
override them. The synthesis pass is where the actual design happens.
