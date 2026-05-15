# CRITIQUE — `loom` decomposition

Contrarian read on the premise of the package, written before reading the
architect's `DESIGN.md`. The goal is independent failure-mode identification,
not adversarial criticism: where I push back, I want the synthesis pass to
weigh both framings.

**Scope (settled, per team-lead clarification 2026-05-15)**: `loom` is a
reusable library peer to `samgria` / `rltrain` / `xptrack`. It is wide:
basis + renderer + targets + tasks + training + diagnostics + plotting.
This critique is about *how* that library is shaped, not whether the
scope is right.

The headline implication of "reusable library" is that **public API
contract matters more than I initially assumed**. Hypothetical second
consumers — a transformer-weight renderer, an RL policy-net renderer,
the FWS programme's own Besov-z-prior diagnostics — need to extend
`Basis`, add an `Encoding`, register a `Target`, register a `Task`
*without forking the package*. The current monolith makes most of these
extension paths hostile. That's §1 below.

## 1. Extensibility — the highest-leverage critique

Walk through each axis a second consumer would have to extend, and look
at the seams the monolith exposes.

### 1a. Adding a new basis

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
threading any new per-basis parameters through `BasisLayer.__init__`,
`BasisBody.__init__`, `make_shared`, and `shared_condition`. Five-file
edit for a one-line mathematical change.

I'd push back on tagged-union-by-string surviving into `loom`. The
honest API for an extensible library is a protocol:

```python
class Basis(eqx.Module, Protocol):
    """A pointwise activation z ↦ a(z), with learnable scalar params."""
    def __call__(self, pre: Array) -> Array: ...
```

Then `SIREN`, `HSIREN`, `WIRE` are concrete `Basis` modules with their
own fields (`omega` for all, `s` for WIRE only). `BasisLayer` becomes
generic over `Basis`. The "pytree uniformity" reason for storing an
unused `s` on every layer evaporates because each basis owns its own
fields.

The second-order benefit: the **slow-param labelling** (line 930,
`SLOW_NAMES = {"omega", "s", "sigma_learnable"}`) is currently a
hardcoded set of attribute names. If a new basis introduces a new
slow-natured parameter (say `kappa` for a sharpness knob), it has to
be added to that string set in a separate file. With a `Basis` protocol,
each basis can declare its own slow params via an `eqx.field` annotation
or a class method `slow_param_names() -> set[str]`. Extension stays local.

### 1b. Adding a new encoding

`LeafConditioning` (lines 142–229) dispatches on `encoding_kind` as a
string. The `fourier_B` array, `sigma_learnable` scalar, and `num_bands`
int are all stored on every instance regardless of which encoding is
active (line 168, 169, 170 — "kept for pytree uniformity"). A new
encoding (say, "wavelet" with its own scale/translation params) must:

1. Edit the `if/elif/else` in `__init__`.
2. Edit the `if/elif/else` in `project`.
3. Add new dummy-init branches for the existing two encodings to keep
   pytree uniformity.
4. Edit the `Encoding` dataclass to include any new params.
5. Edit the constructor helpers (`gaussian_fixed`, `dyadic`, etc.).

I'd push back here too. The `Encoding` value-class (lines 805–822) is
already a discriminated union; lift it from "config dataclass" to
"protocol with a `project` method", and `LeafConditioning` becomes a
composition: `Encoding ∘ HeadProjection ∘ FiLM`. Each `Encoding`
implementation owns its own state. No more dummy `fourier_B` arrays on
dyadic-mode instances.

This is the place where I most expect to disagree with the architect:
the temptation is to lift `Encoding` as a dataclass with `kind: str` and
keep the dispatch. That's *configuration*, not *extension*. A second
consumer who needs a 4th encoding must edit the dispatch — that is, they
must fork.

### 1c. Adding a new target architecture

Targets in the monolith are bare `eqx.Module`s wired into `TaskCfg.template_fn`
(line 1213). That part is actually OK — targets are duck-typed
`eqx.Module` callables, and the renderer's `_is_weight` predicate
(lines 312–322) decides what to virtualise.

But there is one subtle hazard: `_is_weight` is "tensor with ≥2
non-singleton dims" (line 322). That is a *heuristic* tied to the
specific target architectures used so far. A consumer who adds a
target with a rank-1 learned vector that should be virtualised (e.g. a
per-channel scale in a normalisation layer) or with a rank-2 tensor
that should *not* be virtualised (e.g. a learned positional embedding
table) has no override hook.

I'd push back on `_is_weight` being a global module function. It should
be either a strategy passed into `virtualize`, or a per-target
declaration (`target.virtualisable_filter()`). Otherwise the moment
someone adds an attention head, they're patching `_is_weight` and
re-running their tests against an upstream that doesn't know about
their case.

### 1d. Adding a new task

Tasks are `TaskCfg` records (line 1206). Loaders return `TaskData`.
That's clean and extensible — but the registration happens via the
module-level `TASKS = [...]` list (line 1221). A second consumer who
adds CIFAR-100 must either:

- mutate `loom.TASKS` (gross, and breaks the next import order);
- pass their own list explicitly to every harness function;
- define a parallel registry in their consumer code.

The honest pattern is **no module-level registry**. Callers pass a list
of `TaskCfg` instances explicitly. The "convenience" of module-level
`TASKS` is a small surface that costs extension flexibility.

### 1e. Adding a new diagnostic

`_group_grad_norms` (lines 1022–1043) and `_cross_seed_cosine_scalar`
(lines 994–1019) are baked into the inner training scan (lines 1167–
1172). A second consumer who wants to record, say, *per-layer rendered
weight spectral norm* over training has no extension path — they must
fork `train_multi_seed`.

I'd push back on diagnostics being hard-coded into the training loop.
The honest pattern is a list of diagnostic callbacks passed in:

```python
diagnostics: tuple[Diagnostic, ...] = (group_grad_norms, cross_seed_cosine)
```

Each `Diagnostic` is `(params, batch, *) -> jax.Array` and gets stacked
into `RunResult`. The current two diagnostics become the defaults; a
new consumer adds their own without touching the harness.

### 1f. Summary of extensibility critique

The monolith is shaped like a **closed factorial study**: every axis is
a string-tagged enum, every registry is a module-level constant,
every diagnostic is hardcoded. For a one-paper experiment that's fine.
For a reusable library it's an extension-by-fork pattern.

The lift is concrete: replace each `if kind == ...` chain with a
protocol; replace each module-level `*S = [...]` list with a function
parameter; replace `_is_weight` with a target-supplied filter.

## 2. The unstated assumptions

Three framings I expect the architect to default to (orthogonal to §1's
extensibility line) that I would push back on.

### 2a. "The existing module boundaries are correct."

The monolith has banner-comment section dividers (`=== BASES ===`,
`=== TASKS ===`, `=== CONDITIONS ===`, etc.) that read like a
ready-made module list. The temptation is to map each banner to a file.

I would push back on at least two of those boundaries:

- **`BasisLayer` / `BasisBody` / `LeafConditioning` belong together,
  but `LeafConditioning` is *not* a "basis"** — it's the input head +
  FiLM modulation, and it owns three independent encoding modes
  (`none`, `gaussian`, `dyadic`). The "basis" banner conflates two
  things: the inner-MLP activation family and the input encoding. They
  are independently varied in the `Condition` axes (lines 793–832). If
  the package adopts the banner-as-module mapping, encoding ends up in
  the wrong file and the three-axis story falls apart at the API level.
- **`virtualize_standard` vs `virtualize_per_leaf`** look like two
  variants in the monolith (lines 401–427), but their `LeafSlot` /
  `PerLeafSlot` types are almost the same — `PerLeafSlot` is
  `LeafSlot + body`. If the package keeps them parallel, every consumer
  has to type-dispatch on the slot type (which `_render_leaf` already
  does, lines 433–444). A single `Slot` with an optional per-leaf body
  collapses the dispatch — and would *also* make the body-capacity
  question (§3 below) coherently expressible as "what body does this
  slot use?" rather than "is this a shared or per-leaf rendering?".

These boundaries didn't accrete by accident, but they accreted under
pressure to ship — not under pressure to compose cleanly. A
decomposition that takes them as fixed is locking in path-dependence.

### 2b. "It needs a flat `__init__.py` with everything re-exported."

The monolith's consumers all do `import experiment as E` and reach in
by name. About 20–30 symbols across `run_basin_study.py` et al. The
default refactor preserves that flat namespace by re-exporting
everything from `loom.__init__`.

I would push back on `__init__.py` re-exports being the right pattern
at this maturity level. Two failure modes:

- Re-export everything → the package looks flat from outside but is
  structurally nested. Every refactor inside the package risks a
  downstream import break because `__init__.py` is the contract.
- Curate a smaller public surface → downstream scripts break
  immediately, because they reach into things like `E.CONDITIONS` (a
  dict literal edited inline in the monolith) and `E.EVAL_EVERY` (a
  bare module constant).

The honest move is **callers import from submodules directly**:
`from loom.render import render`, `from loom.basis import SIREN`. That
makes the module structure load-bearing — and therefore inspectable —
instead of hiding it behind a flat namespace that pretends to be
stable. The pattern can mature into curated re-exports once the
submodule layout settles. Locking in flat re-exports now locks in the
wrong layout.

(For a library peer to `samgria`/`rltrain`/`xptrack`: it's worth
checking what *those* libraries do. If they use submodule imports, do
the same for consistency; if they use flat re-exports, the case for
matching their convention overrides this argument.)

### 2c. "The three-axis Condition factoring is the natural structure."

The monolith currently structures sweeps as a `Condition` dataclass
parameterised on (basis, encoding, ortho). This factoring landed
**yesterday** (2026-05-15, see CHANGELOG entry "Parameter-free polar
orthogonalisation + three-axis condition API"). It is the most recent
abstraction in the codebase, on a 10-iteration design history.

§11 of the findings doc already names **body capacity** as a fourth
axis the abstraction doesn't own — see §3 of this critique. The
three-axis Condition is overfit to the studies that just ran, and the
next study is pushing on it. Building it into `loom`'s public surface
as the canonical sweep primitive means the library ships with the
abstraction already known to be one axis short.

## 3. Body capacity must be a first-class Condition axis

§11 of the findings doc admits the capacity gap explicitly: the body
MLP has been `SIREN_HIDDEN=24, SIREN_LAYERS=2` throughout, and every
empirical claim is conditional on that one point in capacity space. The
proposed Phase 1 capacity scan (~30 min) is more central to FWS than
any of the §8 next experiments.

I expect the architect to expose body capacity as a config knob on the
"shared body" condition constructor — something like
`shared_condition(..., body_hidden=24, body_layers=2)`. That's
**configurable**, but it's not **a Condition axis** in the sense that
basis/encoding/ortho are. I'd push back on that being enough:

- Capacity should appear in the `Condition` constructor signature on
  equal footing with `basis`, `encoding`, `ortho`. Not as a kwarg with
  a default — as a named, typed axis.
- The condition registry should be **generated**, not enumerated. The
  current `CONDITIONS = {...}` literal (lines 868–910) is 40 lines of
  one-line-per-cell enumeration that breaks the moment you add a fourth
  axis (3 × 4 × 2 × 6 capacity points = 144 conditions). Replace with a
  Cartesian-product builder over named axes.
- The output type of `train_multi_seed` (`RunResult`) should carry the
  full Condition spec, not just a name. Right now (line 1250) the
  binding from result back to its condition is by string key in a dict
  — fragile once axes multiply.

If the architect makes capacity "just another kwarg", the next paper's
capacity-sweep figure will need a bespoke script that bypasses
`CONDITIONS`. That bridge code is the tell that the abstraction is in
the wrong layer (per the `right-layer.md` rule).

A second, sharper version: the **target architecture** is implicitly an
axis too. `DigitsCNN` vs `DigitsDeepCNN` vs `CifarCNN` vs `IrisMLP` all
use the same Conditions. The honest API has five axes
(target × basis × encoding × ortho × capacity), with the registry
generated from axis specs. The `Condition` abstraction currently owns
three of them.

This connects to §1f: if every axis is exposed as a protocol-or-
factory, then the Cartesian-product builder *also* serves the
"hypothetical second consumer adds a new basis" path, because their
new basis is just an additional value on the basis axis. One
abstraction handles both today's capacity sweep and tomorrow's
transformer-renderer extension. That's the natural-convergence signal
worth pursuing.

## 4. Migration cost vs design quality tradeoff

Six runner scripts (`run_basin_study.py`, `run_followup_study.py`,
`run_sigma_probe.py`, `run_cifar_validate.py`, `run_probe.py`,
`build_validation_notebook.py`) all do `import experiment as E` and
reach in by attribute name. They use at minimum: `E.TASKS`,
`E.CONDITIONS`, `E.RunResult`, `E.EVAL_EVERY`, `E.train_multi_seed`,
`E.count_params`, `E.plot`, `E.summarize`, plus target classes
(`E.CifarCNN`, etc.). About 30 distinct symbol-uses across those files.

A clean redesign breaks every one of them. The migration is
straightforward but non-trivial: rename imports, untangle any
encapsulation breaks, re-run experiments to confirm parity, update
artefacts under `runs/*` if result schemas change at all.

I want the contrarian question on the table explicitly: **is now the
right time?** Current state:

- Empirically: a capacity sweep is the next-most-load-bearing experiment.
- Architecturally: the three-axis Condition framing landed yesterday.
- Strategically: there's a paper draft on the horizon (§5b of the
  findings doc), and the headline-efficient configuration is settled.

Two timing options:

1. **Decompose now**, ship `loom` as a wide library covering everything
   from basis through plotting. Migrate all six scripts. Live with the
   fact that the capacity sweep will pressure-test the abstractions
   within weeks.
2. **Run the capacity sweep in the monolith first** (it adds two kwargs
   to `shared_condition`, no API redesign needed), then decompose
   `loom` *after* you have four-axis empirical data instead of three.
   The library that ships is shaped by what you actually need to
   express, not by a snapshot of yesterday's three-axis framing.

I lean toward option 2 — not because the decomposition is wrong, but
because **doing it before the capacity sweep means designing the
sweep-primitive abstraction blind**. A week of delay costs little; a
locked-in three-axis Condition that turns out to be wrong costs every
future user who has to work around it. The user has explicitly preferred
clean mathematical abstractions over convenience wrappers — that
preference is easier to honour with empirical pressure already applied.

If option 1 wins anyway, the §3 demand (capacity-as-axis, generated
registry) becomes load-bearing: it's the only way option 1 doesn't
prematurely freeze.

## 5. What I would do differently — module sketch

A counter-sketch at the module-name level, scoped to the full library.
Order is "smallest, most stable first; largest, most volatile last."
Callers import from submodules directly; `__init__.py` is empty or
declares only `__version__`.

| module | what it owns | stability |
|---|---|---|
| `loom.basis` | `Basis` protocol, `SIREN`, `HSIREN`, `WIRE`, `siren_init` | Stable. Each basis is a concrete class implementing the protocol. |
| `loom.encoding` | `Encoding` protocol, `Identity`, `Gaussian`, `Dyadic`, `LeafConditioning`, `nyquist_sigma` | Stable. Each encoding owns its own state. `LeafConditioning` composes an `Encoding` with the head projection + FiLM. |
| `loom.ortho` | `polar_orthogonalise` | Stable. One function. |
| `loom.slot` | `Slot` (unified — see 2a), `siren_in_dim_for`, `target_init_scale`, `default_weight_filter`, `_normalized_grid` | Virtualisation primitive. Merges `LeafSlot` + `PerLeafSlot`. Weight filter is exposed and overridable. |
| `loom.render` | `virtualize`, `render`, `render_leaf` | The functional render core. Takes a slot tree + optional shared body. |
| `loom.diag` | `Diagnostic` protocol, `group_grad_norms`, `cross_seed_cosine`, `count_params` | Pluggable diagnostics. Consumers add their own implementations. |
| `loom.targets` | `FCHeavyCNN`, `ResidualConvNet`, `IrisMLP` + `DigitsCNN` / `CifarCNN` / `DigitsDeepCNN` / `CifarDeepCNN` partials | Example targets that ship with the library. New targets are user-defined `eqx.Module`s, not subclasses of anything `loom` defines. |
| `loom.tasks` | `TaskData`, `TaskCfg`, loaders (digits, iris, cifar). **No module-level `TASKS` list.** | Example tasks. Callers compose their own list. |
| `loom.sweep` | `Axis`, `cartesian_axes(*axes)`, `Condition` (generated, not enumerated), `RunResult` | The sweep primitive. Generated registry over named axes. Carries the full spec back in `RunResult`. |
| `loom.train` | `train_multi_seed`, `make_optimizer`, `clip_each_leaf`, the slow/main group machinery | Training harness. Takes a `diagnostics` tuple, not a hardcoded set. |
| `loom.plot` | `plot_loss_curves`, `plot_test_acc_curves`, `plot_final_bars`, `plot_grad_dynamics`, `plot_cross_seed_cos`, `summarize` | Decomposed plotters. Each takes a `RunResult` collection + a list of conditions to render. No `DYNAMICS_CONDITION_SUBSET` module-level constant — the caller passes the subset. |

Notable changes from the monolith:

- **`basis` and `encoding` are sibling modules**, not nested under one
  banner. Reflects the data (independent axes), not the file layout.
- **`Slot` is unified** (see §2a). `_render_leaf`'s type dispatch goes
  away.
- **No `TASKS` constant.** Callers pass task lists explicitly. Same
  for the dynamics-subset constant in plotting.
- **`sweep` is a separate module** that owns the axis-product builder
  and `Condition` generation. `train` doesn't import `sweep`; it just
  takes a `Condition` instance.
- **`diag` is pluggable.** Consumers add their own `Diagnostic`
  implementations without touching `train`.
- **`targets` and `tasks` are example collections**, not the registry.
  The library is open to extension because nothing in `loom` enumerates
  them.

## 6. The single biggest risk — concrete failure mode at 3 months

If `loom` ships as a wide library with the three-axis Condition as the
public sweep primitive, here's the specific failure path:

**Month 1**: capacity sweep paper figure. User runs the sweep using
`SIREN_HIDDEN` as a kwarg on `shared_condition`. Works, but the
`CONDITIONS = {...}` enumeration pattern breaks — you can't list 108
entries by hand. User writes a generator script that bypasses
`CONDITIONS` and builds them programmatically. Fine for one figure,
but now there are two parallel registry patterns in the same project.

**Month 2**: FWS Paper-2 transfer-learning angle (§8.6 of findings).
Cross-task σ comparison. The `Condition` doesn't own the task axis, so
a new bespoke script appears that loops over (task, condition) pairs.
Three parallel patterns now.

**Month 3**: hypothetical second consumer — say, a colleague who wants
to apply the renderer to RL policy networks. They want a new
`Basis` ("polynomial-features") and a new `Encoding` (learned per-axis
Fourier coefficients). Both require editing the if/elif/else chains
in `loom.basis` and `loom.encoding`. The colleague forks `loom` rather
than contributing back, because the extension surface is internal-edit-
only. The library has one user.

The throughline: **each new study and each new consumer reveals that
the public API was shaped to the 2026-05-15 snapshot, not to the shape
of the underlying problem**. `loom` and the science diverge. The user
spends weekly debate-time on whether to patch the library or write
around it — the cost the library was meant to *eliminate*.

The path that avoids this: §1 (protocol-based extension), §3
(capacity-as-axis, generated registry), and §4 option 2 (sweep first,
decompose second).

---

## Summary for the synthesis pass

1. **Extensibility is the highest-leverage critique.** Replace each
   tagged-union dispatch with a protocol. `Basis`, `Encoding`,
   `Diagnostic`, the weight-filter. The monolith is shaped for one
   factorial study; a reusable library can't be.
2. **Body capacity must be a first-class axis**, not a kwarg. The
   registry should be generated over named axes via a Cartesian-product
   builder. Target architecture is implicitly a fifth axis.
3. **Module boundaries**: `basis` and `encoding` are siblings, not
   nested. `LeafSlot` + `PerLeafSlot` collapse into one `Slot` with
   optional body.
4. **No flat `__init__.py` re-exports** at this maturity. Callers
   import from submodules. (Override if the peer libraries — `samgria`,
   `rltrain`, `xptrack` — already do flat re-exports; consistency wins
   over this rule.)
5. **Migration timing**: I lean toward running the capacity sweep in
   the monolith first, then decomposing with four-axis empirical
   pressure already applied. If we decompose now, §3 becomes
   load-bearing — generated-registry-over-named-axes is the only thing
   that keeps the design from prematurely freezing.

These are framings to *weigh* against the architect's, not to override
them. The synthesis pass is where the actual design happens.
