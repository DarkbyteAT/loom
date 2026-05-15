# CRITIQUE — `loom` decomposition

Contrarian read on the premise of the package, written before reading the
architect's `DESIGN.md`. The goal is independent failure-mode identification,
not adversarial criticism: where I push back, I want the synthesis pass to
weigh both framings.

## 1. The unstated assumptions

Three framings I expect the architect to default to, each of which I would
push back on.

### 1a. "The renderer is the package."

The most natural reading of "replace the 1415-line monolith with a package
called `loom`" is: take what's already there — bases, conditioning,
virtualisation, render, conditions, training, plot, summarise — split it
across modules, give it an `__init__.py`, ship.

I would push back on that being the right scope. The monolith has at least
three things going on with very different lifecycles:

- **The renderer kernel** (the basis MLPs, conditioning, virtualisation,
  `render`, polar orthogonalisation) — this is the actual scientific
  contribution. Lines ~44–453. Stable for weeks, well-typed, and the only
  part downstream callers (`run_basin_study.py` et al.) reach into
  *concretely* via `E.CONDITIONS` and `E.train_multi_seed`.
- **The condition registry + training harness** (lines ~700–1202) — the
  factorial-experiment driver. This is research-iteration code: the
  three-axis framing landed *yesterday* (2026-05-15). It is on its
  10th-ish design and §11 of the findings doc tells us a 4th axis (body
  capacity) is already pushing on the abstraction.
- **The plotting / summary code** (lines ~1263–1411) — disposable
  artefact generation, plus a `DYNAMICS_CONDITION_SUBSET` constant that
  encodes one specific paper-figure choice.

These three groups have **different maturity, different stability, and
different consumers**. Bundling them into one package with a single
public API hides the fact that the renderer kernel is ~10× more stable
than the harness, and the plotting code shouldn't have a stable API at
all. A package that promises symmetric access to all three is signing a
contract it can't keep, because the harness is going to keep changing.

I'd push back on the package scope being "everything `experiment.py`
currently does" and toward "the renderer kernel only, with the harness
and plotting kept as scripts in the consuming repo." See §4 for what
that looks like.

### 1b. "The existing module boundaries are correct."

The monolith has banner-comment section dividers (`=== BASES ===`,
`=== TASKS ===`, `=== CONDITIONS ===`, etc.) that read like a
ready-made module list. The temptation is to map each banner to a file.

I would push back on at least two of those boundaries:

- **`BasisLayer` / `BasisBody` / `LeafConditioning` belong together, but
  `LeafConditioning` is not a "basis" — it's the input head + FiLM
  modulation, and it owns three independent encoding modes (`none`,
  `gaussian`, `dyadic`). The "basis" banner conflates two things: the
  inner-MLP activation family and the input encoding. They are
  independently varied in the `Condition` axes (see lines ~793–832). If
  the package adopts the banner-as-module mapping, encoding will end up
  in the wrong file and the three-axis story falls apart at the API
  level.
- **`virtualize_standard` vs `virtualize_per_leaf`** look like two
  variants in the monolith (lines ~401–427), but their `LeafSlot` /
  `PerLeafSlot` types are *almost* the same — `PerLeafSlot` is
  `LeafSlot + body`. If the package keeps them parallel, every consumer
  has to type-dispatch on the slot type (which `_render_leaf` already
  does, lines ~433–444). A single `Slot` with an optional per-leaf body
  collapses the dispatch — and would *also* make the body-capacity
  question (§2 below) coherently expressible as "what body does this
  slot use?" rather than "is this a shared or per-leaf rendering?".

These boundaries didn't accrete by accident, but they accreted under
pressure to ship — not under pressure to compose cleanly. A decomposition
that takes them as fixed is locking in path-dependence.

### 1c. "It needs a public API with `__init__.py` re-exports."

The monolith's consumers all do `import experiment as E` and reach in by
name. That import surface is *flat*: `E.CONDITIONS`, `E.TASKS`,
`E.train_multi_seed`, `E.plot`, `E.summarize`, `E.count_params`,
`E.RunResult`, `E.CifarCNN`, etc. — about 20–30 names.

Two failure modes here:

- Re-export everything from `__init__.py` and the package looks
  flat-from-outside but is structurally nested — every refactor inside
  the package still risks a downstream import break, because the
  `__init__.py` is the contract.
- Curate a smaller public surface and downstream scripts break
  immediately, because they reach into `E.CONDITIONS` (a dict literal
  edited inline in the monolith) and `E.EVAL_EVERY` (a bare module
  constant).

I would push back on `__init__.py` re-exports being the right pattern at
this maturity level. The honest move is **no `__init__.py` re-exports
— callers import from submodules directly**: `from loom.render import
render`, `from loom.basis import BasisBody`. That makes the module
structure load-bearing (and therefore inspectable) instead of hiding it
behind a flat namespace that pretends to be stable. The pattern matures
into re-exports once the submodule layout stabilises; locking it in now
locks in the wrong layout.

## 2. The capacity gap — make body capacity a first-class axis

§11 of the findings doc admits the capacity gap explicitly: the body MLP
has been `SIREN_HIDDEN=24, SIREN_LAYERS=2` throughout, and every
empirical claim is conditional on that one point in capacity space. The
proposed Phase 1 capacity scan (~30 min) is more central to FWS than any
of the §8 next experiments.

I expect the architect to expose body capacity as a config knob on the
"shared body" condition constructor — something like
`shared_condition(..., body_hidden=24, body_layers=2)`. That's
**configurable**, but it's not **a Condition axis** in the sense that
basis/encoding/ortho are.

What I'd push back on: the three-axis `Condition` (basis × encoding ×
ortho) is overfit to the studies that just ran. Body capacity is
already pushing into the abstraction as a fourth axis. If the next major
study is the capacity sweep, the API needs to make capacity-as-axis
ergonomic from day one. That means:

- Capacity should appear in the `Condition` constructor signature on
  equal footing with `basis`, `encoding`, `ortho`. Not as a kwarg with
  a default — as a named, typed axis.
- The condition registry should be **generated**, not enumerated. The
  current `CONDITIONS = {...}` literal (lines ~868–910) is 40 lines of
  one-line-per-cell enumeration that breaks the moment you add a fourth
  axis (3 × 4 × 2 × 6 capacity points = 144 conditions). Replace with
  a Cartesian-product builder over named axes.
- The output type of `train_multi_seed` (`RunResult`) should carry the
  full Condition spec, not just a name. Right now (line ~1250) the
  binding from result back to its condition is by string key in a
  dict — fragile once axes multiply.

If the architect makes capacity "just another kwarg", the next paper's
capacity-sweep figure will need a bespoke script that bypasses
`CONDITIONS`. That's the bridge code that signals the abstraction is in
the wrong layer (see the `right-layer.md` rule).

A second, sharper version of this point: the **target architecture** is
also implicitly an axis. `DigitsCNN` vs `DigitsDeepCNN` vs `CifarCNN`
vs `IrisMLP` all use the same Conditions. The four-axis truth is
`target × basis × encoding × ortho × capacity` — five axes. The
`Condition` abstraction currently only owns three of them. The
honest API is one where every axis is first-class and the registry
is generated from axis specs.

## 3. The migration cost vs design quality tradeoff

Six runner scripts (`run_basin_study.py`, `run_followup_study.py`,
`run_sigma_probe.py`, `run_cifar_validate.py`, `run_probe.py`,
`build_validation_notebook.py`) all do `import experiment as E` and
reach in by attribute name. They use at minimum: `E.TASKS`,
`E.CONDITIONS`, `E.RunResult`, `E.EVAL_EVERY`, `E.train_multi_seed`,
`E.count_params`, `E.plot`, `E.summarize`, plus a sprinkling of
target classes (`E.CifarCNN`, etc.). I count ~30 distinct symbol-uses
across those files.

A clean redesign **breaks every one of them**. The migration is
straightforward but non-trivial: rename imports, untangle any
encapsulation breaks, re-run the experiments to confirm parity, update
the artefacts under `runs/*` if the result schemas change at all.

I want to put the contrarian question on the table explicitly: **is now
the right time?** The current state of the programme (per the findings
doc and memory entries) is:

- Empirically: a capacity sweep is the next-most-load-bearing experiment.
- Architecturally: the three-axis Condition framing landed yesterday.
- Strategically: there's a paper draft on the horizon (§5b of the
  findings doc), and the headline-efficient configuration is settled.

Three options, ordered from most-conservative to most-aggressive:

1. **Defer the decomposition.** Add `SIREN_HIDDEN` / `SIREN_LAYERS` as a
   parametrised axis in the monolith, run the capacity sweep, write the
   paper. Decompose *after* the next paper draft, when the abstractions
   have one more empirical pressure test.
2. **Decompose now, but only the kernel.** Extract the renderer kernel
   (basis / conditioning / virtualisation / render / polar) into
   `loom`, leave the harness + plotting + registry in the experiment
   repo. Runners change `import experiment as E` to
   `import loom; import experiment as E` — a small additive migration,
   not a rewrite.
3. **Full decomposition now.** Everything moves. Six scripts rewrite.

Option 2 has the property that the kernel is the part that's *actually*
stable (it hasn't changed materially since 2026-05-12, per the
CHANGELOG), and it's the part that wants reuse outside `experiment/`
(per the `project_fws_framing.md` memory entry — fws-as-a-programme
will want the renderer primitives independent of any specific
experiment). Option 1 has the property that whatever the architect
designs today, the capacity sweep is going to push on it, so designing
*after* the sweep is cheaper.

My contrarian recommendation is **option 2 if the user wants to ship
something now, option 1 if the user is honest about the paper draft
being the actual next deliverable**. Option 3 is the one to push back
on hardest — it has the most cost (6 scripts to rewrite, plus the
existing artefacts under `runs/*`) at the moment when the abstraction
boundaries are still moving fastest.

## 4. What I would do differently — module sketch

If the user picks option 2 or 3 and decomposes the renderer kernel
into a package, my counter-proposal at the module-name level. Order
is "smallest, most stable first; largest, most volatile last."
Critically: **no `__init__.py` re-exports** at this maturity level.
Callers import from submodules directly.

| module | what it owns | scope |
|---|---|---|
| `loom.basis` | `BasisLayer`, `BasisBody`, `siren_init`, `BASIS_KINDS` | Stable. The inner-MLP basis family. |
| `loom.encoding` | `Encoding`, `NO_ENCODING`, `gaussian_*`, `dyadic`, `nyquist_sigma`, `LeafConditioning` | Stable. The input-side encoding, including FiLM. Separated from `basis` because basis and encoding are independently varied — keeping them as siblings reflects the data, not the file structure of the monolith. |
| `loom.ortho` | `polar_orthogonalise` | Stable. One function, one file, one purpose. |
| `loom.slot` | `Slot` (unified — see 1b), `siren_in_dim_for`, `target_init_scale`, `_is_weight`, `_normalized_grid` | The virtualisation primitive. Merges `LeafSlot` + `PerLeafSlot` into a single type with an optional per-leaf body. |
| `loom.render` | `virtualize`, `render`, `render_leaf` | Stable. The functional render core. |
| `loom.diag` | `cross_seed_cosine`, `group_grad_norms`, `count_params` | Diagnostics primitives. Pure functions on rendered pytrees. |

Notably **absent** from the package:

- **`Condition`, `CONDITIONS`, `shared_condition`** — these belong to
  the *experiment harness*, not the renderer. They are
  domain-specific to the basis × encoding × ortho × capacity factorial
  study. Keep in `experiment/` as `experiment/conditions.py`, or
  evolve into a separate `experiment-harness` package later.
- **`train_multi_seed`, `RunResult`, `make_optimizer`,
  `clip_each_leaf`, group-labelling** — training loop infrastructure.
  Belongs in the consumer, not in `loom`. The optimiser group-labelling
  is *specifically* SLOW_NAMES = {omega, s, sigma_learnable}, which is
  knowledge about the renderer's internals, but the rule is
  "tag-by-name during build, label-during-optimise" — that's a
  one-line convention, not a library feature.
- **`plot`, `summarize`, `DYNAMICS_CONDITION_SUBSET`** — paper-figure
  code. Belongs nowhere stable.
- **`TaskCfg`, `TASKS`, `DigitsCNN`, `IrisMLP`, `FCHeavyCNN`,
  `ResidualConvNet`, CIFAR loaders** — task definitions. Belong to the
  consumer; `loom` should be agnostic to what it's rendering weights
  *for*.

The package surface is the **renderer kernel**, ~500 lines, six files.
That's the part that's stable, reusable, and worth promoting. The
remaining ~900 lines of experiment.py stay as scripts in the
experiment repo where they belong.

## 5. The single biggest risk — concrete failure mode at 3 months

If the full decomposition (option 3) ships with the three-axis
`Condition` lifted into the package as part of the public API, here's
what specifically goes wrong:

**The capacity sweep paper figure becomes a bespoke off-API script.**
At month 1, the user runs the capacity sweep using `SIREN_HIDDEN` as
a kwarg on `shared_condition`. It works, but the registry pattern
breaks: you can't enumerate `CONDITIONS` with six capacity points × 18
existing conditions = 108 entries by hand. So the user writes a
generator script that bypasses `CONDITIONS` and builds them
programmatically. Fine for one figure.

At month 2, the FWS Paper-2 transfer-learning angle (§8.6 of the
findings doc) requires cross-task σ comparison. The `Condition` doesn't
own the task axis, so a new bespoke script appears. At month 3, the
Besov-ball z-prior diagnostic (per memory `project_fws_framing.md`,
which says diagnostics are first-class FWS work) needs its own
diagnostics module, but `loom.diag` only knows about rendered weight
pytrees, not latent-space diagnostics. Another bespoke module.

Now we have three off-API scripts and a `loom` whose public surface
encodes a snapshot of the architectural state on 2026-05-15. The
package and the science have diverged: the package is the past, the
scripts are the present. Each new study requires deciding whether to
push it into the package (with API churn) or keep it as a script (with
duplication). The user spends weekly debate-time on that question —
the cost the package was meant to *eliminate*.

The path that avoids this: only put in `loom` the parts that have
already stabilised under multiple studies (the renderer kernel), and
keep the harness/conditions/registry where they are. Re-evaluate after
the capacity-sweep paper draft, when the four-axis framing has been
empirically pressure-tested.

---

## Summary for the synthesis pass

1. Push the package scope **down** to the renderer kernel only —
   ~500 lines, 6 modules — and leave the harness + plotting + condition
   registry in the experiment repo.
2. Body capacity is a first-class Condition axis, not a kwarg. The
   registry should be generated over an axis spec, not enumerated as
   a dict literal.
3. The honest migration is option 2 (extract kernel, keep harness) or
   option 1 (defer entirely until after the paper draft). Option 3 (full
   rewrite of 6 runner scripts) costs the most at the moment the
   abstractions are still moving.
4. No `__init__.py` re-exports yet; callers import from submodules.
5. Merge `LeafSlot` + `PerLeafSlot` into one `Slot` type — the dispatch
   in `_render_leaf` is a tell that they want to be the same thing.

These are framings to *weigh* against the architect's, not to override
them. The synthesis pass is where the actual design happens.
