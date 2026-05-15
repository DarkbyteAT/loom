# `loom` — Public API Contract

> The 1415-line `experiment/experiment.py` monolith renders neural-network
> weights from a small INR. This document specifies the public contract of the
> `loom` package that replaces it. The contract is opinionated, decided,
> and final unless the open questions in §7 say otherwise.

## Library positioning

`loom` is a **reusable library**, peer to `samgria` (SAM variants),
`rltrain` (RL algorithms), and `xptrack` (experiment tracking). The
specific experiment in `experiment/experiment.py` is **one consumer** of
`loom`, not its defining use case.

Future FWS-adjacent work — transformer weight renderers, RL policy-network
renderers, diffusion U-Net renderers, etc. — should `import loom`, write
~20 lines of glue (a new target class, a new `TaskCfg`, a custom
`ConditionRegistry`), and run. They should *not* fork the monolith.

That framing has three load-bearing consequences for this contract:

1. **Reference targets and loaders ship inside `loom`** as built-ins
   (`FCHeavyCNN`, `load_digits`, …). They are *examples of the
   contract*, not "experimental fixtures". Adding a new target or
   dataset is one file + one registry line; the built-ins demonstrate
   the pattern.
2. **`Condition`, `Encoding`, `Basis` are the library's three core value
   types**. They must be top-level importable: `from loom import Condition,
   SIREN, Gaussian` works. Submodule paths exist for narrower imports
   but the top-level surface is the primary advertised contract.
3. **Runner scripts are NOT part of `loom`.** The eight scripts currently
   in `experiment/` (`run_probe.py`, `run_basin_study.py`, …) become
   *consumers* of `loom` — they either stay in `experiment/` as a
   downstream project, or move under `examples/` in this repo as
   demonstrations. Either way, `pip install loom` does not install them.
   See §5.

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
| `loom.renderer` | `LeafSlot`, `PerLeafSlot`, `SharedRenderer`, `PerLeafRenderer`, `DirectRenderer`, `Renderer` protocol, `virtualize`, `render` | `basis`, `conditioning`, `ortho`, `config` |
| `loom.condition` | `Topology`, `Condition`, `ConditionRegistry` | `basis`, `encoding`, `renderer`, `config` |
| `loom.targets` | `FCHeavyCNN`, `IrisMLP`, `ResidualConvNet`, `DigitsCNN`, `CifarCNN`, `DigitsDeepCNN`, `CifarDeepCNN` | — |
| `loom.tasks` | `TaskData`, `TaskCfg`, `TaskRegistry`, `load_digits`, `load_iris`, `load_cifar10` | `targets`, `config` |
| `loom.training` | `RunResult`, `train_multi_seed`, `make_optimizer`, `clip_each_leaf` | `condition`, `tasks`, `diagnostics`, `config` |
| `loom.diagnostics` | `cross_seed_cosine`, `group_grad_norms`, gradient-group labelling | `renderer` |
| `loom.metrics` | `cross_entropy`, `accuracy`, `count_params` | — |
| `loom.plotting` | `plot_run`, `plot_dynamics`, `summarize` | `condition`, `training` |
| `loom.presets` | `default_conditions()`, `default_tasks()`, `default_renderer_config()` | every other module |

`loom/__init__.py` re-exports the same names the monolith currently surfaces
to scripts, so a runner can do `from loom import train_multi_seed, ...`.
Everything is also importable from the submodule for code that wants the
narrower import.

### Why `presets` rather than module-level defaults

This is the single most important shape change. The monolith has
`SIREN_HIDDEN`, `OMEGA_FIRST`, `S_INIT`, `NUM_SEEDS`, `DIGITS_STEPS`, …
sitting at module scope. They are mutated implicitly by every probe script
and cannot be varied per-run without monkey-patching. The capacity-sweep
experiment (findings §11) requires `SIREN_HIDDEN`/`SIREN_LAYERS` to be a
function argument, not a global. `loom.presets` is the only place a default
value lives. Everywhere else a value is taken explicitly.

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
container, `presets.default_conditions(cfg)` is where the 21-row table
lives**. The monolith's module-level `CONDITIONS: dict[str, Condition]`
dict gets *mutated* by `run_probe.py` via try/finally hacks; that's the
single biggest source of action-at-a-distance bugs in the codebase. The
registry is immutable; `select` and `filter` return new registries.

### 2.8 `loom.targets`

```python
__all__ = ["FCHeavyCNN", "IrisMLP", "ResidualConvNet",
           "DigitsCNN", "CifarCNN", "DigitsDeepCNN", "CifarDeepCNN",
           "Target"]

class Target(Protocol):
    """A target network template. Any `eqx.Module` constructible with a
    `key=...` kwarg satisfies this protocol. There is no required base
    class — this is structural typing, so user-defined targets compose
    without subclassing anything from `loom`."""
    def __init__(self, *, key: PRNGKeyArray, **kwargs: Any): ...
    def __call__(self, x: Array) -> Array: ...

class FCHeavyCNN(eqx.Module):
    """Reference FC-heavy target. The renderer's hardest case (FC index-
    permutation symmetry fights the coord-smoothness prior). Parameterised
    on in_channels / in_size / conv_stride / hidden / n_classes."""
class IrisMLP(eqx.Module):
    """Reference tiny MLP target — rank-2 weights only."""
class ResidualConvNet(eqx.Module):
    """Reference all-conv target (SiLU residual blocks). Every learnable
    weight is rank-4; the renderer's easiest case."""

DigitsCNN = partial(FCHeavyCNN, in_channels=1, in_size=8, conv_stride=1)
CifarCNN  = partial(FCHeavyCNN, in_channels=3, in_size=32, conv_stride=2)
DigitsDeepCNN = partial(ResidualConvNet, in_channels=1, stem_stride=1)
CifarDeepCNN  = partial(ResidualConvNet, in_channels=3, stem_stride=2)
```

**Extensibility contract.** Adding a new target architecture is one file
+ one registry line:

```python
# in user code (e.g. loom_transformer/targets.py)
class SmallTransformer(eqx.Module):
    def __init__(self, *, key, d_model=64, n_heads=4, depth=2): ...
    def __call__(self, x): ...

# in user code (e.g. loom_transformer/conditions.py)
SmallTransformerPreset = partial(SmallTransformer, d_model=64, n_heads=4)
```

The target satisfies the `Target` protocol structurally. It plugs into
`TaskCfg.template_fn` directly — no registration with `loom`, no
inheritance, no metaclass. The built-in targets are themselves just
instances of this protocol; nothing about them is privileged.

The reference targets stay in `loom.targets` because (a) they're the
canonical examples of the protocol, (b) the monolith's experiments are
the immediate consumer, (c) downstream FWS work on, e.g., a transformer
renderer can extend / subclass them without re-implementing the bias-
isn't-a-weight invariant. Class shapes and partials are unchanged from
the monolith.

### 2.9 `loom.tasks`

```python
__all__ = ["TaskData", "TaskCfg", "TaskRegistry",
           "load_digits", "load_iris", "load_cifar10"]

@dataclass(frozen=True, slots=True)
class TaskData:
    """A loaded dataset split. The contract: x is anything `template_fn`
    accepts as input, y is integer labels. Any user-defined loader that
    returns this shape plugs into `TaskCfg` directly."""
    xs_train: Float[Array, "n ..."]
    ys_train: Int[Array, "n"]
    xs_test: Float[Array, "m ..."]
    ys_test: Int[Array, "m"]
    name: str

@dataclass(frozen=True, slots=True)
class TaskCfg:
    """The PRIMARY contract for 'a thing loom can train on'.

    A `TaskCfg` is a quintuple of (a name, a target template factory, a
    data loader, a step budget, a batch size, a learning rate). Anything
    callable that fits these slots works — the built-in `load_digits`
    etc. are conveniences, not the contract.

    Typical user code constructs `TaskCfg` directly:

        TaskCfg("my-task", template_fn=MyTransformer,
                loader=my_loader, num_steps=2000, batch_size=64, lr=1e-3)
    """
    name: str
    template_fn: Callable[..., eqx.Module]   # any Target-protocol factory
    loader: Callable[[], TaskData]            # any () -> TaskData
    num_steps: int
    batch_size: int
    lr: float

class TaskRegistry(Mapping[str, TaskCfg]):
    """Same shape as ConditionRegistry — ordered, immutable, .select / .filter."""
    ...

# Built-in convenience loaders. None of these are privileged — they're
# reference implementations of the (() -> TaskData) contract.
def load_digits(seed: int = 0) -> TaskData: ...
def load_iris(seed: int = 0) -> TaskData: ...
def load_cifar10(seed: int = 0, *, n_train: int = 1500,
                 n_test: int = 400) -> TaskData: ...
```

**Extensibility contract.** Loading a new dataset is one function:

```python
# in user code
def load_my_dataset(seed: int = 0) -> TaskData:
    xs, ys = ...   # whatever (numpy, torch, fsspec, huggingface) you want
    return TaskData(xs_train=..., ys_train=..., xs_test=..., ys_test=...,
                    name="my-dataset")

task = TaskCfg("my-task", template_fn=MyTarget, loader=load_my_dataset,
               num_steps=2000, batch_size=64, lr=1e-3)
res  = train_multi_seed(my_cond, task, ...)
```

No registration, no plugin system, no `loom`-side opt-in. The
`Callable[[], TaskData]` shape *is* the contract.

The monolith's module-level `TASKS = [...]` list is replaced by the
`TaskRegistry` returned from `loom.presets.default_tasks()` — that
registry is the *built-in* set; downstream projects compose their own
the same way (`TaskRegistry([digits_cfg, my_cfg])`). CIFAR caching
becomes a `functools.lru_cache` on `load_cifar10`, not a module-level
`_CIFAR_CACHE` dict.

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

### 2.12 `loom.presets`

```python
__all__ = ["default_renderer_config", "default_training_config",
           "default_conditions", "default_tasks"]

def default_renderer_config() -> RendererConfig: ...     # the 24/2/6.0/1.0/3.0 defaults
def default_training_config() -> TrainingConfig: ...     # 5 seeds, eval_every=25, ...
def default_conditions(cfg: RendererConfig | None = None) -> ConditionRegistry: ...
def default_tasks() -> TaskRegistry: ...
```

`default_conditions` is the canonical home for the 21-row condition table.
A script that wants a probe subset writes
`default_conditions().select(["direct", "shared-si+ff-pi"])` — no module
mutation.

---

## 3. Killing the globals — explicit treatment

| Monolith global | Replacement | Where it lives |
|---|---|---|
| `SIREN_HIDDEN`, `SIREN_LAYERS`, `OMEGA_FIRST`, `OMEGA_HIDDEN`, `S_INIT` | `RendererConfig` fields | `loom.config`; default in `loom.presets.default_renderer_config()` |
| `NUM_SEEDS`, `EVAL_EVERY`, `LR_MULT`, `SLOW_NAMES`, `PER_LEAF_GRAD_CAP` | `TrainingConfig` fields | `loom.config`; default in `loom.presets.default_training_config()` |
| `DIGITS_STEPS`, `IRIS_STEPS`, `DIGITS_BATCH`, `IRIS_BATCH` | Per-task `TaskCfg` fields | `loom.presets.default_tasks()` |
| `CONDITIONS: dict[str, Condition]` (mutable!) | `ConditionRegistry` (immutable Mapping) | `loom.presets.default_conditions()` |
| `TASKS: list[TaskCfg]` | `TaskRegistry` | `loom.presets.default_tasks()` |
| `BASIS_KINDS = ("siren", "hsiren", "wire")` | Subclass set in `loom.basis` | `loom.basis` |
| `MASTER_SEED = 42` | `Final[int]` in `loom.config` | `loom.config` |
| `_CIFAR_CACHE: dict[str, TaskData]` | `functools.lru_cache(maxsize=4)` on `load_cifar10` | `loom.tasks` |
| `GROUP_ORDER = ("main", "slow")` | Static method on `TrainingConfig` | `loom.config` |
| `DYNAMICS_CONDITION_SUBSET` | Callers pass it via `.select(...)` | gone from `loom`; lives in scripts |

The probe scripts' `try: CONDITIONS["x"] = ...; finally: CONDITIONS["x"] = old`
pattern is structurally impossible after the refactor. The capacity sweep
becomes:

```python
for hidden in (12, 16, 24, 48, 96):
    cfg = default_renderer_config().with_(hidden=hidden)
    cond = Condition.shared("shared-si+ff-nyq+ortho", "#fb8072",
                            basis=SIREN(), encoding=GaussianPerLeaf(nyquist_sigma),
                            ortho=True)
    res[hidden] = train_multi_seed(cond, digits_task,
                                   renderer_cfg=cfg, training_cfg=tcfg)
```

That's the §11 capacity-sweep requirement satisfied at the API level.

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

**Recommended: Option B' — port-then-shim, single PR per ported module,
delete the monolith in one final PR, then split runner scripts between
`loom/examples/` (demos) and `experiment/` (the active FWS consumer).**

Why not Option A (port everything alongside, kill `experiment/` in one
PR): the monolith is 1415 lines with eight runner scripts, an analysis
notebook, and a CHANGELOG that traces specific functions by name. One
mega-PR has too much surface area to review carefully, and bisecting a
regression against a single squash commit is painful.

Why not strict Option B (keep `experiment/` as a thin shim forever):
shims that stay around become load-bearing. The `finish-what-you-break`
rule says you must commit to deleting them.

The chosen strategy:

1. **Port modules bottom-up**, in dependency order: `config`, `basis`,
   `encoding`, `ortho`, `conditioning`, `renderer`, `targets`, `tasks`,
   `condition`, `presets`, `diagnostics`, `metrics`, `training`,
   `plotting`. Each module ships with its own tests (functions and
   classes have unit-level tests; renderer composition has integration
   tests against monolith output on `master_seed=42`).
2. **`experiment/experiment.py` becomes a re-export shim after each
   module ports**, so existing runner scripts (`run_probe.py`,
   `run_basin_study.py`, etc.) keep working *unchanged*. The shim's job
   is two-fold: route old names to new locations, and route old
   condition-key strings to new ones.
3. **Numerical equivalence test on `master_seed=42`** runs after every
   port: the new code, given the same `Condition` and `TaskCfg`, must
   produce a bit-identical `RunResult` to the monolith. This is the
   convergence criterion for each port.
4. **Final library PR deletes `experiment/experiment.py` and the
   shim.** At this point `loom` is a self-contained library installable
   via `pip install loom`. The CHANGELOG entry for this PR is an
   inventory of what moved where.
5. **Post-port: split runner scripts into two destinations.**

   - **Two or three become canonical `loom/examples/`** — small,
     self-contained demos that exercise the library's surface area
     (e.g. `examples/single_condition.py` — train one condition on
     digits; `examples/capacity_sweep.py` — the §11 sweep; perhaps
     `examples/custom_target.py` — show how downstream code adds a new
     target). These are the docs-as-code surface for new users.
   - **The rest stay in `experiment/`** as the active FWS-specific
     consumer of `loom`. That directory becomes a peer project:
     `experiment/pyproject.toml` lists `loom` as a dependency, the
     runner scripts import from `loom`, the npz files and research
     notes live alongside. `samgria`, `rltrain`, future
     `loom_transformer`, etc. follow the same shape — they're
     downstream consumers, not library code.

   The split criterion is reusability: if a script demonstrates a
   library feature in ≤ 100 lines and would help a new user, it goes
   in `examples/`. If a script encodes a specific FWS experimental
   design (basin study with n=30, σ sweeps, CIFAR validation), it
   stays in `experiment/`. Default to `experiment/` when in doubt.

The migration owns one Trello card per ported module (so the wave-
execution rules apply: narrow ports first, broader ports last,
`condition` and `training` last because they pull every other module
together). Steps 4 and 5 are separate cards.

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
- **The eight-condition naming convention** (`shared-si`, `shared-hs`,
  `shared-w`, `+ortho`, `+ff-pi`, …). New code uses the same key
  strings so historical `.npz` files stay readable.
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
   appears in the condition registry. The alternative — make every
   `Encoding` instance carry its sampled `B` even at the registry level
   — would require deferring all registry construction until a PRNG is
   available, which makes `default_conditions()` take a `key` argument.
   I'd rather pay the explicit `materialise` call.

5. **`pandas` dependency for `summarize`.** Returning a DataFrame
   instead of printing is cleaner, but pulls pandas in. Alternative:
   return a list of `dataclass`-typed rows and let the caller format.
   I lean DataFrame; flag for confirmation.

6. **Lazy CIFAR-10 import.** The monolith imports torchvision lazily
   inside `_load_cifar10_arrays`. I want to keep that — torchvision is a
   heavy dep and digits/iris users shouldn't pay for it. Confirm the
   `loom.tasks.load_cifar10` signature is allowed to do `import
   torchvision` inside its body.

7. **`Basis.init_params` returning a tuple of scalars.** Slightly
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

### 8a. The 30-second example — using the built-in target

```python
import math
from loom import (
    Condition, SIREN, GaussianPerLeaf, nyquist_sigma,
    TaskCfg, DigitsCNN, load_digits,
    train_multi_seed, RendererConfig, TrainingConfig,
)

cond = Condition.shared(
    "shared-si+ff-nyq+ortho", color="#fb8072",
    basis=SIREN(omega_init=6.0),
    encoding=GaussianPerLeaf(nyquist_sigma),
    ortho=True,
)

task = TaskCfg(
    name="digits-cnn", template_fn=DigitsCNN, loader=load_digits,
    num_steps=1500, batch_size=128, lr=3e-3,
)

res = train_multi_seed(
    cond, task,
    renderer_cfg=RendererConfig(hidden=48),   # capacity knob (§11)
    training_cfg=TrainingConfig(),
)

print(f"{res.n_params=}, test_acc={res.eval_curves[:, -1, 1].mean():.3f}")
```

### 8b. The 20-line glue example — bringing your own target + dataset

The library-positioning claim ("FWS-adjacent work writes ~20 lines of
glue, not a fork"). This is what those 20 lines look like:

```python
import equinox as eqx
from jaxtyping import Array, PRNGKeyArray
from loom import (Condition, SIREN, Gaussian, TaskCfg, TaskData,
                  train_multi_seed, RendererConfig, TrainingConfig)

class TinyTransformer(eqx.Module):
    # ... any eqx.Module accepting `key=...`. Satisfies loom's Target protocol.
    def __init__(self, *, key: PRNGKeyArray): ...
    def __call__(self, x: Array) -> Array: ...

def load_my_data(seed: int = 0) -> TaskData:
    # ... return TaskData(xs_train=..., ys_train=..., xs_test=..., ys_test=...,
    #                     name="my-dataset")
    ...

cond = Condition.shared("transformer-siren+ff-pi", color="#1b9e77",
                        basis=SIREN(), encoding=Gaussian(sigma=math.pi))
task = TaskCfg("my-task", TinyTransformer, load_my_data,
               num_steps=2000, batch_size=64, lr=1e-3)
res = train_multi_seed(cond, task,
                       renderer_cfg=RendererConfig(),
                       training_cfg=TrainingConfig())
```

Zero subclassing of anything in `loom`. The target is a plain
`eqx.Module`; the loader is a plain function. The library only owns the
renderer, the condition algebra, and the training loop — everything
else plugs in by protocol.

No globals. No mutations. No `kind=` strings. One config object per concern.
Body capacity is a function argument. Adding a new basis/encoding/ortho
combination is one `Condition.shared(...)` line. Adding a new target or
dataset is a class and a function.

---

**End of contract.**
