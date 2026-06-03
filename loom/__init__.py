"""loom — a substrate for re-parameterising JAX pytrees by arbitrary functions.

See `docs/PHILOSOPHY.md` for the design verdict, principles, and contract.
Public surface: one primitive (`render`) and one exception family
(`RenderError` + `ShapeMismatch` + `DTypeMismatch`).
"""

from loom.errors import DTypeMismatch, RenderError, ShapeMismatch
from loom.render import render


__all__ = [
    "DTypeMismatch",
    "RenderError",
    "ShapeMismatch",
    "render",
]
