# CLAUDE.md

@AGENTS.md

## Project Context

A substrate for re-parameterising JAX pytrees by the output of arbitrary functions, in a way that is scannable and optimisable. Public surface: `loom.render(P, f, params)` plus `RenderFn` (Protocol) and the `RenderError`/`ShapeMismatch`/`DTypeMismatch` exception family. INRs from sibling library `ondes` are the canonical instantiation but are not baked in. See `docs/PHILOSOPHY.md` for principles and contract guarantees.
