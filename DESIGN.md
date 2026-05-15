# `loom` — Public API Contract

> The 1415-line `experiment/experiment.py` monolith renders neural-network
> weights from a small INR. This document specifies the public contract of the
> `loom` package that replaces it. The contract is opinionated, decided,
> and final unless the open questions in §7 say otherwise.

## Library positioning

`loom` is the **machinery**, not the **instantiations**. Same scope rule
as the peer libraries:

- `rltrain` ships RL algorithms, not env-specific deps (no `minatar`,
  no `ple`); users plug a gymnasium env from downstream scripts.
- `samgria` ships SAM/ASAM/ImplicitMAML primitives, not training loops.
- `xptrack` ships the tracker, not a particular dataset of experiments.
- **`loom` ships the renderer, the condition algebra, and the training
  loop primitives — not concrete targets, not dataset loaders, not a
  pre-populated condition registry.**

### What `loom` owns

- `Basis` ABC + `SIREN`, `HSIREN`, `WIRE` subclasses
- Renderer machinery — virtualisation, `render`, `polar_orthogonalise`,
  FiLM, `LeafSlot`, `PerLeafSlot`, `LeafConditioning`
- `Encoding` ABC + `Identity`, `Gaussian`, `GaussianPerLeaf`,
  `LearnedGaussian`, `Dyadic`, `nyquist_sigma`
- `Condition` value type composing the three axes (basis × encoding ×
  ortho) + `ConditionRegistry` container
- The **`Target` protocol** and the **`TaskCfg`/`TaskData` types** —
  these are *contracts*, not concrete implementations
- Training primitives: `train_multi_seed`, `RunResult`, optimiser-group
  machinery, `clip_each_leaf`
- Diagnostics primitives: cross-seed cosine, group grad norms
- Plotting helpers that consume `RunResult` and produce the standard
  curves/bars

### What `loom` does NOT own

- ❌ **Dataset loaders.** `load_digits` (sklearn), `load_iris`
  (sklearn), `load_cifar10` (torchvision) are sklearn/torchvision-
  coupled and demonstration code. They do not live in the package.
- ❌ **Concrete target architectures.** `FCHeavyCNN`, `IrisMLP`,
  `ResidualConvNet` and their `DigitsCNN` / `CifarCNN` /
  `DigitsDeepCNN` / `CifarDeepCNN` partials are tied to specific input
  shapes and classification budgets. They are demonstration code.
- ❌ **The pre-populated condition table.** The 25-row `CONDITIONS`
  dict (`direct`, `shared-si+ff-pi`, `shared-si+ff-nyq+ortho`, …) is a
  specific experimental design. `loom` ships the `ConditionRegistry`
  *type* and the `Condition.shared(...)` *constructor*; the canned
  registry is demo code.
- ❌ **Runner scripts.** `run_basin_study.py`, `run_followup_study.py`,
  `run_sigma_probe.py`, `run_cifar_validate.py`, `run_probe.py` are
  downstream consumers.

### Where the demo code lives

See §5 for the migration recommendation between (a) `loom/examples/`
in this repo (uninstalled subdirectory) and (b) a separate
`experiment/` repo that depends on `loom`. Spoiler: I recommend (b).

### Why this scope is the right one

The "FWS-adjacent work writes ~20 lines of glue" claim still stands —
the glue is: define an `eqx.Module` target, define a `() -> TaskData`
loader, build a `Condition`, call `train_multi_seed`. The reason `loom`
must *not* own targets/loaders is exactly the §11 capacity-gap insight:
every empirical claim in the monolith is conditional on a specific
target architecture and dataset. Promoting one combination into the
library would smuggle those design choices into every downstream
consumer. The peer libraries (`rltrain`, `samgria`) learned this lesson
already — `loom` inherits it.

The `Condition` / `Encoding` / `Basis` triple are still top-level
importable: `from loom import Condition, SIREN, Gaussian` works.

## 0. Anchoring vocabulary

The mathematical object loom realises is

```
W_eff(shape) = polar?( out_scale * ( body( γ(coords(shape)) ; FiLM ) - mean ) )
```

where

- `coords(shape)` is a normalised grid over the weight tensor's index axes,
- `γ` is an **Encoding** ∈ { Identity, Gaussian(σ), Dyadic(L) } applied
  per-coord,
- `body` is a **Basis** body MLP ∈ { SIREN(ω), HSIREN(ω), WIRE(ω, s) },
- `FiLM` is an optional per-leaf modulation tensor inside the body,
- `polar?` is the optional Björck–Bowie semi-orthogonal projection (rank-2
  weights only),
- `out_scale` is the Kaiming-style target init magnitude (suppressed when
  `polar?` is active).

A **Condition** is a `(Basis, Encoding, ortho: bool)` triple plus a
**Topology** ∈ { Direct, Shared, PerLeaf } that says where the body lives.
A **Task** is a `(template_fn, loader, budget)` triple. A **Run** is what
you get from training one Condition on one Task across `num_seeds` seeds.

That decomposition is the spine of the module map.

---

## 1. Module map

Each `loom/<module>.py` owns exactly one concept from §0. Cross-module
imports always go *down* this list, never up. There are no circular
imports.

| module | owns | depends on |
|---|---|---|
| `loom.config` | `RendererConfig`, `TrainingConfig`, deterministic seeds | — |
| `loom.basis` | `Basis` ABC, `SIREN`, `HSIREN`, `WIRE`, `BasisLayer`, `BasisBody`, `siren_init` | `config` |
| `loom.encoding` | `Encoding` ABC, `Identity`, `Gaussian`, `GaussianPerLeaf`, `LearnedGaussian`, `Dyadic`, `nyquist_sigma` | — |
| `loom.ortho` | `polar_orthogonalise` | — |
| `loom.conditioning` | `LeafConditioning` (the per-leaf head + FiLM) | `encoding`, `config` |
| `loom.renderer` | `LeafSlot`, `PerLeafSlot`, `SharedRenderer`, `PerLeafRenderer`, `DirectRenderer`, `Renderer` protocol, `virtualize`, `render`, `is_weight`, `siren_in_dim_for`, `target_init_scale` | `basis`, `conditioning`, `ortho`, `config` |
| `loom.condition` | `Topology`, `Condition`, `ConditionRegistry` | `basis`, `encoding`, `renderer`, `config` |
| `loom.target_protocol` | `Target` Protocol — what shape an `eqx.Module` must have to be renderable (rank-2/4 weights, bias detection rules) | — |
| `loom.task_protocol` | `TaskData`, `TaskCfg`, `TaskRegistry`, `LossFn` Protocol | `target_protocol` |
| `loom.training` | `RunResult`, `train_multi_seed`, `make_optimizer`, `clip_each_leaf` | `condition`, `task_protocol`, `diagnostics`, `config` |
| `loom.diagnostics` | `cross_seed_cosine`, `group_grad_norms`, gradient-group labelling | `renderer` |
| `loom.metrics` | `cross_entropy`, `accuracy`, `count_params` | — |
| `loom.plotting` | `plot_run`, `plot_dynamics`, `summarize` (take `ConditionRegistry` + `RunResult` dict — no canned data) | `condition`, `training` |

What's gone vs the previous draft: `loom.targets` (the FCHeavyCNN /
IrisMLP / ResidualConvNet module), `loom.tasks` (the load_digits /
load_cifar10 module), and `loom.presets` (the canned-condition /
canned-task factory module). They migrate to the demo layer — see §5.

`loom/__init__.py` exposes the value types and the training entry point
at top level: `from loom import (Basis, SIREN, HSIREN, WIRE, Encoding,
Identity, Gaussian, GaussianPerLeaf, LearnedGaussian, Dyadic,
nyquist_sigma, Condition, ConditionRegistry, Topology, RendererConfig,
TrainingConfig, TaskCfg, TaskData, train_multi_seed, RunResult, ...)`.
Submodule paths exist for narrower imports.

### Why no `presets` module

The monolith has `SIREN_HIDDEN`, `OMEGA_FIRST`, `S_INIT`, `NUM_SEEDS`,
`DIGITS_STEPS`, …, *and* `CONDITIONS`, *and* `TASKS` all sitting at
module scope. The previous draft consolidated the numeric defaults into
a `RendererConfig` / `TrainingConfig` dataclass and the canned content
into a `loom.presets` factory module. The narrower scope decision means
**the canned content is gone** — `default_conditions()`, `default_tasks()`
are demo concerns, not library concerns. The numeric defaults are now
just the dataclass field defaults on `RendererConfig` / `TrainingConfig`
themselves; no separate module is needed.

The capacity-sweep requirement (findings §11) is satisfied at the
library level by `RendererConfig(hidden=...)` being a constructor
argument with a sensible default. Downstream code does
`RendererConfig(hidden=48)` directly; no `loom.presets`
indirection.

---

## 2. Public API per module

For every name in `__all__`, one-line signature + one-line description.
Type annotations are part of the contract.

### 2.1 `loom.config`

```python
__all__ = ["RendererConfig", "TrainingConfig", "MASTER_SEED"]

MASTER_SEED: Final[int] = 42  # deterministic root key for all sweeps

@dataclass(frozen=True, slots=True)
class RendererConfig:
    """All renderer hyperparameters. Threaded through constructors; never read from globals."""
    hidden: int = 24
    layers: int = 2
    omega_first: float = 6.0
    omega_hidden: float = 1.0
    s_init: float = 3.0              # WIRE envelope width init
    polar_iters: int = 12            # Björck–Bowie iteration count
    fourier_features: int = 16       # default Gaussian-encoding num_freqs

    def with_(self, **overrides) -> "RendererConfig": ...   # immutable update

@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """All training-loop hyperparameters."""
    num_seeds: int = 5
    eval_every: int = 25
    per_leaf_grad_cap: float = 1.0
    slow_lr_mult: float = 0.1
    slow_param_names: frozenset[str] = frozenset({"omega", "s", "sigma_learnable"})
```

Rationale: every empirical claim in the codebase is conditional on these
numbers. They MUST be visible in run artefacts, not buried in source.
`RendererConfig` is what the capacity sweep iterates over.

### 2.2 `loom.basis`

```python
__all__ = ["Basis", "SIREN", "HSIREN", "WIRE", "BasisLayer", "BasisBody",
           "siren_init"]

class Basis(eqx.Module, ABC):
    """Activation family for a body layer. Subclasses parameterise their own learnables."""
    @abstractmethod
    def activate(self, pre: Array, *, omega: Array, s: Array) -> Array: ...
    @abstractmethod
    def init_params(self, *, key) -> tuple[Array, Array]:
        """Return (omega_init_value, s_init_value) as scalar Arrays."""

class SIREN(Basis):
    """sin(ω · pre).  (Sitzmann et al. 2020.)"""
    omega_init: float

class HSIREN(Basis):
    """sin(ω · sinh(pre)).  (Cai & Pan 2024.)"""
    omega_init: float

class WIRE(Basis):
    """cos(ω · pre) · exp(-(s·pre)²).  (Saragadam et al. 2023.)"""
    omega_init: float
    s_init: float

class BasisLayer(eqx.Module):
    """One hidden layer: y = Basis.activate(γ ⊙ (W x + b) + β, omega, s)."""
    def __init__(self, in_dim: int, out_dim: int, basis: Basis, *,
                 is_first: bool, key: PRNGKeyArray): ...

class BasisBody(eqx.Module):
    """Stack of BasisLayers + scalar readout. The 'body' in W = body(γ(coords))."""
    def __init__(self, in_dim: int, basis: Basis, cfg: RendererConfig,
                 *, key: PRNGKeyArray): ...
    def __call__(self, x: Array, film: Array | None = None) -> Array: ...

def siren_init(in_dim: int, out_dim: int, omega: float, *, is_first: bool,
               key: PRNGKeyArray) -> tuple[Array, Array]: ...
```

Decision: **`Basis` is an Equinox ABC with concrete subclasses, not a
`kind: Literal["siren","hsiren","wire"]` string discriminator**. The
monolith's `if self.kind == "siren": …` chain is a closed enum. Making it
polymorphic kills the dispatch chain *and* makes WIRE's extra `s_init`
argument live on WIRE, not on every layer. `BasisLayer` still has `omega`
and `s` as learnable arrays — the *parameter pytree* is uniform across
bases (so `vmap` works) but the *type* discriminates dispatch.

### 2.3 `loom.encoding`

```python
__all__ = ["Encoding", "Identity", "Gaussian", "GaussianPerLeaf",
           "LearnedGaussian", "Dyadic", "nyquist_sigma"]

class Encoding(eqx.Module, ABC):
    """γ(coord). Owns its own learnable state (or lack of it) and out-dim."""
    @abstractmethod
    def out_dim(self, rank: int) -> int: ...
    @abstractmethod
    def encode(self, coord: Float[Array, "rank"]) -> Float[Array, "out"]: ...
    @abstractmethod
    def materialise(self, rank: int, *, key: PRNGKeyArray) -> "Encoding":
        """Build the concrete per-leaf instance (samples B for Gaussian etc.)."""

class Identity(Encoding):
    """γ(coord) = coord."""

class Gaussian(Encoding):
    """γ(coord) = [cos(2πBc), sin(2πBc)] with B ~ N(0, σ²)."""
    sigma: float
    num_freqs: int = 16

class GaussianPerLeaf(Encoding):
    """Gaussian whose σ is computed per-leaf from the weight shape (e.g. Nyquist)."""
    sigma_rule: Callable[[tuple[int, ...]], float]
    num_freqs: int = 16

class LearnedGaussian(Encoding):
    """Gaussian whose σ is a learnable scalar (init sigma_init, tagged 'slow')."""
    sigma_init: float = math.pi
    num_freqs: int = 16

class Dyadic(Encoding):
    """NeRF-style sin/cos ladder, 2^k π · coord_i for k = 0..L-1."""
    L: int = 4

def nyquist_sigma(shape: tuple[int, ...]) -> float:
    """σ = max(1, (max(shape) - 1) / 4) — Tancik's index-grid Nyquist."""
```

Decision: **`Encoding` is an ABC with concrete subclasses, not a
`@dataclass(frozen=True)` discriminator union**. The monolith's `Encoding`
dataclass has `kind`, `sigma`, `sigma_from_shape`, `learn_sigma`, `num_bands`
all coexisting at type level — three of them are unused for any given
`kind`. The constructors `gaussian_fixed`, `gaussian_learn`, `dyadic` exist
because the dataclass itself is awkward to construct. Subclasses replace
the constructor functions and remove the dead fields per variant.

`materialise(rank, key)` is the explicit factory that produces a per-leaf
instance with the random `B` matrix sampled. The user-facing `Gaussian(σ=π)`
is a *spec*; the rendered slot holds a `Gaussian` with `B` filled in.
Splitting spec from materialised instance is what kills the "module-level
CONDITIONS dict gets mutated by probes" anti-pattern: every materialise is
explicit.

### 2.4 `loom.ortho`

```python
__all__ = ["polar_orthogonalise"]

def polar_orthogonalise(W: Float[Array, "m n"], *,
                        n_iter: int = 12) -> Float[Array, "m n"]:
    """Björck–Bowie polar iteration: X ← X(3I − XᵀX)/2.
    Parameter-free; preserves autograd through matmuls only. Frobenius-
    normalised input guarantees convergence (‖W‖_F ≥ ‖W‖₂)."""
```

Decision: stays as a free function. No state, no class, no config. The
algorithm is locked.

### 2.5 `loom.conditioning`

```python
__all__ = ["LeafConditioning"]

class LeafConditioning(eqx.Module):
    """Per-leaf input head + FiLM.

    Owns one Encoding instance, one (head_W, head_b), one FiLM tensor.
    The 'three encoding modes via kind string' anti-pattern in the
    monolith is gone — the Encoding does its own dispatch."""
    encoding: Encoding
    head_W: Float[Array, "in_dim proj_in"]
    head_b: Float[Array, "in_dim"]
    film: Float[Array, "L two_h"]
    rank: int = eqx.field(static=True)

    def __init__(self, *, rank: int, siren_in_dim: int, encoding: Encoding,
                 cfg: RendererConfig, key: PRNGKeyArray): ...
    def project(self, coord: Float[Array, "rank"]) -> Float[Array, "siren_in_dim"]: ...
```

### 2.6 `loom.renderer`

```python
__all__ = ["LeafSlot", "PerLeafSlot", "Renderer", "SharedRenderer",
           "PerLeafRenderer", "DirectRenderer", "virtualize", "render",
           "siren_in_dim_for", "target_init_scale", "is_weight"]

class LeafSlot(eqx.Module):
    """A virtualised weight leaf, rendered by an *external* shared body."""
    cond: LeafConditioning
    in_dim: int = eqx.field(static=True)
    out_scale: float = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    ortho: bool = eqx.field(static=True)

class PerLeafSlot(eqx.Module):
    """A virtualised weight leaf with its own private body."""
    body: BasisBody
    cond: LeafConditioning
    in_dim: int = eqx.field(static=True)
    out_scale: float = eqx.field(static=True)
    shape: tuple[int, ...] = eqx.field(static=True)
    ortho: bool = eqx.field(static=True)

class Renderer(Protocol):
    """Forward-pass protocol: given a target template + a batch, produce logits."""
    def init(self, template: eqx.Module, key: PRNGKeyArray) -> eqx.Module: ...
    def forward(self, params: eqx.Module, x: Array) -> Array: ...

class DirectRenderer(eqx.Module):
    """Identity renderer: trains the target directly. (The 'direct' baseline.)"""

class SharedRenderer(eqx.Module):
    """One body shared across all rendered leaves. Stores body + virt pytree + use_film."""
    body: BasisBody
    virt: eqx.Module
    use_film: bool = eqx.field(static=True)

class PerLeafRenderer(eqx.Module):
    """A per-leaf body per virtualised leaf."""
    virt: eqx.Module
    use_film: bool = eqx.field(static=True)

def virtualize(template: eqx.Module, *, basis: Basis, encoding: Encoding,
               ortho: bool, topology: Topology, cfg: RendererConfig,
               key: PRNGKeyArray) -> Renderer:
    """Walk the target template, replace weight leaves with slots, return Renderer."""

def render(renderer: Renderer) -> eqx.Module:
    """Materialise the rendered weight pytree (the actual target with concrete weights)."""

def siren_in_dim_for(rank: int) -> int: ...        # max(4*rank, 4)
def target_init_scale(shape: tuple[int, ...]) -> float: ...
def is_weight(x: Any) -> bool: ...
```

Decision: **one public `virtualize` function dispatching on `Topology`,
returning a `Renderer`.** The monolith has `make_shared`, `make_per_leaf`,
and the implicit direct-baseline path; that's three call sites that all
mean "build the runtime renderer for this condition". Collapsing them to
one entry point lets `Condition.materialise` be one line.

### 2.7 `loom.condition`

```python
__all__ = ["Topology", "Condition", "ConditionRegistry"]

class Topology(enum.Enum):
    DIRECT = "direct"
    SHARED = "shared"
    PER_LEAF = "per_leaf"

@dataclass(frozen=True, slots=True)
class Condition:
    """A `(basis, encoding, ortho, topology)` quadruple + presentation metadata.

    Two ways to build a Condition:
        Condition(name=..., color=..., basis=SIREN(), encoding=Gaussian(sigma=π))
        Condition.shared("name", color, basis=..., encoding=..., ortho=...)
        Condition.per_leaf("name", color, basis=...)
        Condition.direct("name", color)
    """
    name: str
    color: str
    basis: Basis
    encoding: Encoding
    ortho: bool = False
    topology: Topology = Topology.SHARED

    def materialise(self, template: eqx.Module, cfg: RendererConfig,
                    *, key: PRNGKeyArray) -> Renderer:
        return virtualize(template, basis=self.basis, encoding=self.encoding,
                          ortho=self.ortho, topology=self.topology,
                          cfg=cfg, key=key)

    @classmethod
    def direct(cls, name: str, color: str) -> "Condition": ...
    @classmethod
    def shared(cls, name: str, color: str, *, basis: Basis,
               encoding: Encoding = Identity(), ortho: bool = False) -> "Condition": ...
    @classmethod
    def per_leaf(cls, name: str, color: str, *, basis: Basis,
                 ortho: bool = False) -> "Condition": ...

class ConditionRegistry(Mapping[str, Condition]):
    """An ordered, immutable mapping. Pass one to train_multi_seed / plot."""
    def __init__(self, items: Iterable[Condition]): ...
    def select(self, keys: Iterable[str]) -> "ConditionRegistry": ...
    def filter(self, predicate: Callable[[Condition], bool]) -> "ConditionRegistry": ...
```

Decision: **`Condition` is a value object, `ConditionRegistry` is the
container, and the canned 25-row table is demo code, not library code.**
The monolith's module-level `CONDITIONS: dict[str, Condition]` dict
gets *mutated* by `run_probe.py` via try/finally hacks; that's the
single biggest source of action-at-a-distance bugs in the codebase. The
registry is immutable; `select` and `filter` return new registries.

The pre-populated registry (`direct`, `per-leaf-si`, `shared-si`,
`shared-hs`, `shared-w`, `shared-si+ortho`, …, `shared-w+ff-nyq+ortho`)
encodes a specific experimental design — basis × encoding × ortho
choices motivated by FWS hypotheses about coord-smoothness and
spectral structure. It lives in the demo project (likely
`experiment/conditions.py`), not in `loom`. `loom` provides the
constructor `Condition.shared(...)` and the container
`ConditionRegistry([...])`; building the canned set is one call to each
in the demo project.

### 2.8 `loom.target_protocol`

The library defines *what shape a target must have*, not *what targets
exist*. Concrete targets live downstream.

```python
__all__ = ["Target", "TemplateFn"]

class Target(Protocol):
    """The structural contract for a target network.

    Any `eqx.Module` that:
      - is constructed via `cls(*, key: PRNGKeyArray, ...)`
      - is callable as `model(x: Array) -> Array`
      - exposes its weights as a pytree where `is_weight(leaf)` correctly
        identifies the renderable tensors (≥ 2 non-singleton dims;
        rank-1 biases and rank-3 (C, 1, 1) broadcast biases are skipped)
    satisfies this protocol. No subclassing required; structural typing.
    """
    def __init__(self, *, key: PRNGKeyArray) -> None: ...
    def __call__(self, x: Array) -> Array: ...

TemplateFn = Callable[..., Target]
"""A template factory: usually a class, or a `functools.partial` of one.
`TaskCfg.template_fn` is of this type."""
```

**Extensibility contract.** A downstream consumer defines its target
once, plugs it into `TaskCfg.template_fn`:

```python
# in user code — e.g. experiment/targets.py or loom_transformer/targets.py
import equinox as eqx, jax
from loom import Target  # protocol — for static checks only

class MyTarget(eqx.Module):
    def __init__(self, *, key): ...
    def __call__(self, x): ...

# This is the contract. No registration, no plugin system, no
# inheritance from anything in loom.
```

The bias-detection rules (rank-1 biases and `(C, 1, 1)` broadcast
biases are *not* rendered) are properties of `is_weight` in
`loom.renderer`, not of the `Target` protocol. The protocol just says
"be a callable equinox module". The renderer's `is_weight` walks the
target's pytree and decides per-leaf what's renderable.

### 2.9 `loom.task_protocol`

The library defines *what a task looks like*, not *which tasks exist*.
No dataset loaders, no scikit-learn dependency, no torchvision.

```python
__all__ = ["TaskData", "TaskCfg", "TaskRegistry", "Loader", "LossFn"]

@dataclass(frozen=True, slots=True)
class TaskData:
    """A loaded dataset split. The contract: x is anything the target's
    `__call__` accepts, y is integer class labels (for classification).
    Any user-defined loader that returns this shape satisfies it."""
    xs_train: Float[Array, "n ..."]
    ys_train: Int[Array, "n"]
    xs_test: Float[Array, "m ..."]
    ys_test: Int[Array, "m"]
    name: str

Loader = Callable[[], TaskData]
"""A dataset loader — any callable returning `TaskData`. The library
does not ship loaders; downstream code writes a `Loader` for whatever
dataset it cares about (sklearn, torchvision, huggingface, fsspec,
synthetic, …)."""

LossFn = Callable[[Array, Array], Array]
"""(logits, labels) → scalar loss. Defaults to cross-entropy in the
training loop, but `TaskCfg` can override per task (e.g. MSE for
regression targets)."""

@dataclass(frozen=True, slots=True)
class TaskCfg:
    """THE primary contract for 'a thing loom can train on'.

    A `TaskCfg` bundles:
      - `name`         — display string
      - `template_fn`  — a `TemplateFn` (target factory)
      - `loader`       — a `Loader` (zero-arg → `TaskData`)
      - `num_steps`    — training budget in optimiser steps
      - `batch_size`   — minibatch size
      - `lr`           — base learning rate (multi_transform group rates
                         apply on top)
      - `loss_fn`      — optional `LossFn` (defaults to cross-entropy)

    Downstream code constructs `TaskCfg` directly. The library does not
    own any concrete instance.
    """
    name: str
    template_fn: TemplateFn
    loader: Loader
    num_steps: int
    batch_size: int
    lr: float
    loss_fn: LossFn | None = None   # None → use library default (CE)

class TaskRegistry(Mapping[str, TaskCfg]):
    """Same shape as ConditionRegistry — ordered, immutable,
    `.select(keys)` / `.filter(pred)` return new registries.
    Downstream code builds its own — `loom` ships no instance."""
    def __init__(self, items: Iterable[TaskCfg]): ...
    def select(self, keys: Iterable[str]) -> "TaskRegistry": ...
    def filter(self, predicate: Callable[[TaskCfg], bool]) -> "TaskRegistry": ...
```

**Extensibility contract.** Loading a dataset is a function the user
writes:

```python
# in user code — e.g. experiment/tasks.py
from sklearn import datasets
from sklearn.model_selection import train_test_split
from loom import TaskData, TaskCfg
from my_targets import MyTarget   # user-defined, see §2.8

def load_my_dataset(seed: int = 0) -> TaskData:
    d = datasets.load_digits()
    X, y = d.data.astype("float32") / 16.0, d.target.astype("int32")
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=seed, stratify=y)
    return TaskData(jnp.asarray(Xtr), jnp.asarray(ytr),
                    jnp.asarray(Xte), jnp.asarray(yte), "my-digits")

task = TaskCfg("my-task", template_fn=MyTarget, loader=load_my_dataset,
               num_steps=2000, batch_size=64, lr=1e-3)
```

No `pip install loom` opt-in for sklearn, torchvision, or any specific
dataset. The `Callable[[], TaskData]` shape *is* the contract — and
that contract makes zero assumptions about how the data was loaded.

The monolith's `_CIFAR_CACHE` dict and the `TASKS = [...]` list both
disappear from the library entirely. The CIFAR cache becomes a
`functools.lru_cache` on whatever loader the demo project writes —
that's not a `loom` concern.

### 2.10 `loom.training`

```python
__all__ = ["RunResult", "train_multi_seed", "make_optimizer", "clip_each_leaf"]

@dataclass(frozen=True, slots=True)
class RunResult:
    losses: np.ndarray       # (seeds, steps)
    eval_curves: np.ndarray  # (seeds, n_eval, 2)
    grad_norms: np.ndarray   # (n_eval, groups)
    cross_cos: np.ndarray    # (n_eval,)
    n_params: int

def train_multi_seed(
    condition: Condition,
    task: TaskCfg,
    *,
    renderer_cfg: RendererConfig,
    training_cfg: TrainingConfig,
    master_seed: int = MASTER_SEED,
) -> RunResult:
    """Multi-seed training run. Loads the task lazily, builds the renderer
    per-seed, trains, returns aggregated metrics."""

def make_optimizer(lr: float, training_cfg: TrainingConfig
                   ) -> optax.GradientTransformation: ...
def clip_each_leaf(max_norm: float) -> optax.GradientTransformation: ...
```

Note the signature change: the monolith's `train_multi_seed` takes
`(cond, template_fn, task, num_steps, batch_size, lr, num_seeds)` —
seven positional / mixed args. The new one takes a `Condition`, a
`TaskCfg` (which carries `template_fn`, `num_steps`, `batch_size`, `lr`),
and the two config dataclasses. Five args, all named, all keyword-only
after `task`. Capacity-sweep code reads
`train_multi_seed(cond, task, renderer_cfg=cfg.with_(hidden=48), ...)`.

### 2.11 `loom.diagnostics`, `loom.metrics`, `loom.plotting`

Self-explanatory; signatures preserve the monolith's behaviour. Notable:
`plotting.plot_run` and `plotting.summarize` take a `ConditionRegistry`
explicitly, never the module-level `CONDITIONS`. `summarize` returns a
`pandas.DataFrame` instead of just printing — printing is the caller's
job.

### 2.12 (removed)

The previous draft had a `loom.presets` module owning
`default_renderer_config()` / `default_training_config()` /
`default_conditions()` / `default_tasks()`. The narrower-scope decision
drops it entirely:

- Numeric defaults are dataclass field defaults on `RendererConfig` /
  `TrainingConfig`. Callers write `RendererConfig()` or
  `RendererConfig(hidden=48)`; there is no factory function to wrap
  this.
- `default_conditions()` and `default_tasks()` were canned content
  (specific FWS experimental design + sklearn/torchvision-coupled
  data). They become demo-project concerns. The downstream
  `experiment/` repo writes its own `conditions.py` and `tasks.py`
  that build a `ConditionRegistry` / `TaskRegistry`.

---

## 3. Killing the globals — explicit treatment

| Monolith global | Replacement | Where it lives |
|---|---|---|
| `SIREN_HIDDEN`, `SIREN_LAYERS`, `OMEGA_FIRST`, `OMEGA_HIDDEN`, `S_INIT` | `RendererConfig` field defaults | `loom.config` |
| `NUM_SEEDS`, `EVAL_EVERY`, `LR_MULT`, `SLOW_NAMES`, `PER_LEAF_GRAD_CAP` | `TrainingConfig` field defaults | `loom.config` |
| `DIGITS_STEPS`, `IRIS_STEPS`, `DIGITS_BATCH`, `IRIS_BATCH` | Per-task `TaskCfg` fields, set by the consumer | downstream demo project (`experiment/`) |
| `CONDITIONS: dict[str, Condition]` (mutable!) | `ConditionRegistry` *type* in `loom`; the canned 25-row instance lives downstream | type: `loom.condition`; instance: `experiment/conditions.py` |
| `TASKS: list[TaskCfg]` | `TaskRegistry` *type* in `loom`; the canned instance lives downstream | type: `loom.task_protocol`; instance: `experiment/tasks.py` |
| `BASIS_KINDS = ("siren", "hsiren", "wire")` | Subclass set in `loom.basis` | `loom.basis` |
| `MASTER_SEED = 42` | `Final[int]` in `loom.config` | `loom.config` |
| `_CIFAR_CACHE: dict[str, TaskData]` | `functools.lru_cache` on the downstream loader | `experiment/tasks.py` |
| `GROUP_ORDER = ("main", "slow")` | Static method on `TrainingConfig` | `loom.config` |
| `DYNAMICS_CONDITION_SUBSET` | Callers pass it via `.select(...)` | downstream scripts |
| `FCHeavyCNN`, `IrisMLP`, `ResidualConvNet`, `DigitsCNN`, … | Concrete `eqx.Module`s satisfying `Target` protocol | downstream demo project (`experiment/targets.py`) |
| `load_digits`, `load_iris`, `load_cifar10` | Concrete loaders satisfying the `Loader` protocol | downstream demo project (`experiment/tasks.py`) |

The probe scripts' `try: CONDITIONS["x"] = ...; finally: CONDITIONS["x"] = old`
pattern is structurally impossible after the refactor (the registry is
immutable). The capacity sweep — pure-library code — reads:

```python
from loom import (Condition, ConditionRegistry, SIREN, GaussianPerLeaf,
                  RendererConfig, TrainingConfig, train_multi_seed)
from experiment.targets import DigitsCNN
from experiment.tasks   import digits_task           # a TaskCfg
from experiment.conditions import nyquist_sigma      # or loom.encoding

results = {}
for hidden in (12, 16, 24, 48, 96):
    cfg = RendererConfig(hidden=hidden)
    cond = Condition.shared(
        "shared-si+ff-nyq+ortho", "#fb8072",
        basis=SIREN(), encoding=GaussianPerLeaf(nyquist_sigma),
        ortho=True,
    )
    results[hidden] = train_multi_seed(
        cond, digits_task,
        renderer_cfg=cfg, training_cfg=TrainingConfig(),
    )
```

That's the §11 capacity-sweep requirement, satisfied at the API level,
with `loom` carrying the renderer machinery and the demo project
carrying the target + dataset choices.

---

## 4. Mathematics-first ergonomics

The user explicitly asked for code that reads like the maths. Here is the
shape of a typical call, side-by-side with the formula:

```
Math:  W_eff = polar( body_siren( γ_gauss(coords) ; FiLM ) - mean )

Code:  cond = Condition.shared(
           "shared-si+ff-pi+ortho", "#bc80bd",
           basis=SIREN(omega_init=6.0),
           encoding=Gaussian(sigma=math.pi),
           ortho=True,
       )
```

The three axes — basis, encoding, ortho — are *three constructor
arguments*, each a value object. No `kind=` strings, no `sigma=None`
sentinel, no `if encoding.kind == "gaussian"` dispatch in user code.
Adding a new basis is one class, not a new `kind` branch in seven
functions.

### Why this composition is correct (not just pretty)

It matches the *natural convergence* test in `rules/design`:

- `Basis` has no business knowing what `Encoding` is doing — the body MLP
  operates on whatever vector its input head produced. Decoupling them
  removes the cross-axis "WIRE+ff-pi", "H-SIREN+dyadic" combinatorial
  explosion from the dispatch code.
- `polar?` doesn't care which basis or encoding produced `W` — it's a
  pure projection on rank-2 weights. Making it a boolean instead of a
  Basis-aware path is the right abstraction.
- The slot is rank-aware (ortho only on rank-2); the renderer is not.
  Pushing that check down to slot construction lets `render` be a plain
  `tree_map`.

This is what §0 means by "the abstraction is aligned with the problem's
structure." Each axis lives in its own module and composes without
adapter code.

---

## 5. Migration strategy

**Recommended: port machinery into `loom`; then split the monolith's
demo content between `experiment/` (downstream consumer) and
`loom/examples/` (tiny canonical demos). Recommend the `experiment/`
side become a separate repository depending on `loom`.**

Why not "port everything alongside, kill `experiment/` in one PR": the
monolith is 1415 lines with eight runner scripts, an analysis notebook,
and a CHANGELOG that traces specific functions by name. One mega-PR
has too much surface area to review carefully, and bisecting a
regression against a single squash commit is painful.

Why not "keep `experiment/` as a thin shim forever": shims that stay
around become load-bearing. The `finish-what-you-break` rule says you
must commit to deleting them.

The chosen strategy:

1. **Port machinery modules bottom-up**, in dependency order: `config`,
   `basis`, `encoding`, `ortho`, `conditioning`, `renderer`,
   `target_protocol`, `task_protocol`, `condition`, `diagnostics`,
   `metrics`, `training`, `plotting`. **Concrete targets, loaders, and
   the canned condition table are NOT ported into `loom`**; they go
   straight to the demo project. Each ported module ships with its own
   tests; renderer composition has integration tests against monolith
   output on `master_seed=42`.

2. **`experiment/experiment.py` becomes a re-export shim after each
   module ports**, so existing runner scripts (`run_probe.py`,
   `run_basin_study.py`, etc.) keep working *unchanged* during the
   migration. The shim's job is two-fold: route old names to new
   locations (e.g. `from experiment import polar_orthogonalise` →
   `from loom import polar_orthogonalise`), and host the still-
   downstream content (target classes, loaders, the canned
   `CONDITIONS` registry) without claiming any of it lives in `loom`.

3. **Numerical equivalence test on `master_seed=42`** runs after every
   machinery port: the new `loom` code, given the same `Condition` /
   `TaskCfg` (loaded via the shim's still-monolithic loaders), must
   produce a bit-identical `RunResult` to the pre-port monolith. This
   is the convergence criterion for each machinery port.

4. **Migration-finale PR** — delete `experiment/experiment.py` (the
   shim), and split its contents into:

   - **`experiment/targets.py`** — `FCHeavyCNN`, `IrisMLP`,
     `ResidualConvNet`, `DigitsCNN`, `CifarCNN`, `DigitsDeepCNN`,
     `CifarDeepCNN`.
   - **`experiment/tasks.py`** — `load_digits`, `load_iris`,
     `load_cifar10` (with their `functools.lru_cache`), and the
     canned `TaskRegistry`.
   - **`experiment/conditions.py`** — the 25-row pre-populated
     `ConditionRegistry`, plus `nyquist_sigma` IF it stays domain-
     specific (open question: it's currently in `loom.encoding`; it
     may belong in the demo).
   - **`experiment/run_*.py`** — runners imported from above + `loom`,
     unchanged in essence.

   At this point `pip install loom` installs only the machinery.
   `experiment/` is a sibling that depends on `loom`.

5. **Recommendation: make `experiment/` a separate repository.**
   The peer libraries (`samgria`, `rltrain`, `xptrack`) live in
   separate repos; the demo project for `loom` should too. Reasons:

   - **Dependency boundary clarity.** `experiment/` depends on
     sklearn and torchvision; `loom` depends only on JAX/Equinox/
     Optax. Keeping them in one repo invites accidental imports
     across the boundary. A separate repo's `pyproject.toml`
     enforces this physically.
   - **Independent release cadence.** Empirical experiments move
     fast (a new study every few days); the library should move
     slowly (semver, breakage matters). Different repos = different
     release cycles.
   - **Future-proofing the multi-consumer claim.** When a
     `loom_transformer` or `loom_rl_policy` consumer appears, the
     existing pattern is already "depend on `loom`, live in your
     own repo". One demo-in-loom-repo would be the odd one out.

   Alternative: keep `experiment/` as a subdirectory in this repo
   marked uninstalled (`pyproject.toml` excludes it from the package
   wheel). Lower migration cost, but it's the wrong long-term shape
   per the reasons above. Default to a separate repo.

6. **Tiny canonical `loom/examples/`.** Two or three scripts, each
   < 100 lines, exercising the library surface with a *trivial*
   target and *synthetic* data (random tensors with a known
   factorisation, say). These exist for documentation, not for
   experiments. Examples:

   - `examples/render_a_weight.py` — virtualise a single `eqx.nn.Linear`,
     render it, show the shape and norm.
   - `examples/capacity_sweep_synthetic.py` — the §11 sweep on a
     toy MLP with random data; demonstrates the API.
   - `examples/custom_basis.py` — define a new `Basis` subclass.

   These are *not* the digits/iris/CIFAR studies. Those move to
   `experiment/`.

The migration owns one Trello card per ported module (so the wave-
execution rules apply: narrow ports first, broader ports last,
`condition` and `training` last because they pull every other module
together). Steps 4, 5, and 6 are separate cards.

---

## 6. Things explicitly NOT changed

Locked-in invariants that this design preserves without comment:

- **JAX / Equinox / Optax dependency stack.** No PyTorch, no Flax,
  no Haiku, no NNX migration.
- **`BasisBody` is an `eqx.Module`** — the field shape (`layers`,
  `readout_W`, `readout_b`, `hidden_dim`, `num_hidden_layers`, `kind`)
  is preserved modulo the `kind` → `basis: Basis` change.
- **`polar_orthogonalise` algorithm.** Björck–Bowie 12-iteration loop,
  Frobenius-normalised input, no SVD fallback. The 2026-05-15 changelog
  entry is the spec.
- **`master_seed=42` determinism contract.** Identical seed → identical
  `RunResult` byte-for-byte. The whole capacity-sweep methodology
  depends on this.
- **`is_weight` rule.** A leaf with ≥ 2 non-singleton dims is a weight;
  everything else stays a direct trainable param.
- **The "subtract per-coord mean before applying `out_scale`" step.**
  It's load-bearing for matching the symmetric-uniform init in
  expectation.
- **The `(seeds, steps)` / `(seeds, n_eval, 2)` / `(n_eval, groups)` /
  `(n_eval,)` array shapes in `RunResult`.** Downstream analysis code
  depends on these.
- **`scan_tqdm` on the outer block-scan** for progress reporting.

---

## 7. Open questions

These are decisions I'm flagging rather than making — small enough that
either answer is fine, large enough that the wrong call costs a follow-up
refactor.

1. **`Topology` enum vs. polymorphic `Renderer` constructors.** I picked
   the enum because the three topologies share a `(template → renderer)`
   shape and the dispatch table is small. The alternative is
   `SharedRenderer.build(...)` / `PerLeafRenderer.build(...)` /
   `DirectRenderer.build(...)` and no enum at all. The enum is simpler
   for `Condition.shared(...)` factories; the polymorphic form is purer.
   Either works.

2. **Should `Condition` carry `topology` as a field, or split into
   `DirectCondition` / `SharedCondition` / `PerLeafCondition`
   subclasses?** I chose one dataclass with a `Topology` field because
   it lets `ConditionRegistry` be `Mapping[str, Condition]` (one type),
   and because the topology axis doesn't have axis-specific extra
   parameters worth a subclass. If we end up adding per-topology config
   (e.g. per-leaf body-size override), revisit.

3. **`color` as a `Condition` field at all.** Presentation concern
   bleeding into the data model. The cleaner split is `Condition`
   (pure) + a separate `ConditionTheme` mapping in `plotting`. I left
   it on `Condition` to keep the migration delta small, but I'd accept
   a counter-proposal to split.

4. **`Encoding.materialise(rank, key)` vs `Encoding.__init__`
   accepting `rank` and `key` directly.** I picked `materialise` because
   `Gaussian(sigma=π)` reads like a spec (no PRNG) and the spec is what
   appears in a downstream condition registry. The alternative — make
   every `Encoding` instance carry its sampled `B` even at the spec
   level — would require deferring all registry construction until a
   PRNG is available, which would force every downstream
   `conditions.py` to take a `key` argument at module import time.
   I'd rather pay the explicit `materialise` call.

5. **`pandas` dependency for `summarize`.** Returning a DataFrame
   instead of printing is cleaner, but pulls pandas in. Alternative:
   return a list of `dataclass`-typed rows and let the caller format.
   I lean DataFrame; flag for confirmation. (If we say no, `plotting`
   stays pure-matplotlib and the summary is a list of dataclasses.)

6. **Where does `nyquist_sigma` live?** It's a small pure function
   `(shape: tuple[int, ...]) -> float` that gives a Tancik-style σ for
   Gaussian encoding from a weight tensor's shape. Two homes:
   (a) `loom.encoding` — it parameterises a `GaussianPerLeaf` and is
   the canonical example of a `sigma_rule`; (b) downstream demo —
   it's domain-specific to the FWS index-grid analysis. I lean (a)
   because it's pure, tiny, and demonstrates the `sigma_rule`
   contract; the alternative argument is that "Tancik-style σ from
   Nyquist analysis" is an FWS-flavoured *choice*, not machinery.

7. **Should `loom` preserve the monolith's condition-key naming
   convention** (`shared-si`, `shared-hs+ff-pi`, …) as documented
   guidance for downstream registries? Historical `.npz` files use
   those keys. The library itself doesn't care — it never instantiates
   those names — but the migration matters: the demo project
   (`experiment/`) needs to keep them so old artefacts stay readable.
   I'd treat this as a downstream convention, mentioned in `loom`'s
   examples but not enforced.

8. **`Basis.init_params` returning a tuple of scalars.** Slightly
   awkward — WIRE has both `omega_init` and `s_init`, SIREN/HSIREN only
   use `omega`. The pytree-uniformity constraint (so vmap works across
   `BasisLayer`s of different bases — though we don't currently mix
   them) means we always carry both. Either we keep the pytree-uniform
   invariant (and `Basis.init_params` always returns `(omega_init,
   s_init)`), or we drop it (and the runtime check that a `WIRE` layer
   has its `s` becomes a structural type guarantee). I leaned uniform;
   open to either.

---

## 8. Sanity-check — what fits in a single page

Here's the entire user-facing surface of `loom` as a runnable script.
If this isn't readable in 30 seconds the design has failed.

### 8a. The 20-line glue example — bringing your own target + dataset

The library-positioning claim ("downstream work writes ~20 lines of
glue, not a fork"). Since `loom` ships no concrete targets or loaders,
this *is* the minimum example — there is no "built-in" path.

```python
import math
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray
from sklearn.datasets import load_digits as sk_load_digits
from sklearn.model_selection import train_test_split

from loom import (
    Condition, SIREN, Gaussian,
    TaskCfg, TaskData,
    train_multi_seed, RendererConfig, TrainingConfig,
)

class MyTarget(eqx.Module):
    # any eqx.Module taking key=...; satisfies loom's Target protocol.
    def __init__(self, *, key: PRNGKeyArray): ...
    def __call__(self, x: Array) -> Array: ...

def my_loader(seed: int = 0) -> TaskData:
    d = sk_load_digits()
    X = (d.data.astype("float32") / 16.0).reshape(-1, 1, 8, 8)
    Xtr, Xte, ytr, yte = train_test_split(X, d.target.astype("int32"),
                                          test_size=0.2, random_state=seed,
                                          stratify=d.target)
    return TaskData(jnp.asarray(Xtr), jnp.asarray(ytr),
                    jnp.asarray(Xte), jnp.asarray(yte), "digits")

cond = Condition.shared("siren+ff-pi", color="#1b9e77",
                        basis=SIREN(omega_init=6.0),
                        encoding=Gaussian(sigma=math.pi))
task = TaskCfg("digits", MyTarget, my_loader,
               num_steps=1500, batch_size=128, lr=3e-3)
res  = train_multi_seed(cond, task,
                        renderer_cfg=RendererConfig(hidden=48),  # capacity knob (§11)
                        training_cfg=TrainingConfig())

print(f"{res.n_params=}, test_acc={res.eval_curves[:, -1, 1].mean():.3f}")
```

Zero subclassing of anything in `loom`. The target is a plain
`eqx.Module`; the loader is a plain function. sklearn is a downstream
dependency, not a library one. The library only owns the renderer, the
condition algebra, the training loop, and the protocol types — every
domain-specific choice is in user code.

### 8b. The capacity sweep — pure-library code

```python
from loom import (Condition, SIREN, GaussianPerLeaf,
                  RendererConfig, TrainingConfig, train_multi_seed)
from loom.encoding import nyquist_sigma            # (or moved downstream — see Q6)
# downstream:
from experiment.targets import DigitsCNN
from experiment.tasks   import digits_task          # a TaskCfg

results = {}
for hidden in (12, 16, 24, 48, 96):
    cond = Condition.shared(
        "shared-si+ff-nyq+ortho", color="#fb8072",
        basis=SIREN(), encoding=GaussianPerLeaf(nyquist_sigma),
        ortho=True,
    )
    results[hidden] = train_multi_seed(
        cond, digits_task,
        renderer_cfg=RendererConfig(hidden=hidden),
        training_cfg=TrainingConfig(),
    )
```

No globals. No mutations. No `kind=` strings. One config object per concern.
Body capacity is a function argument. Adding a new basis/encoding/ortho
combination is one `Condition.shared(...)` line. Adding a new target or
dataset is one `eqx.Module` and one `Callable[[], TaskData]`, both in
downstream code. The library never sees them.

---

**End of contract.**
