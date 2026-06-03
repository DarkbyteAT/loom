"""loom — a substrate for re-parameterising JAX pytrees by arbitrary functions.

See `docs/PHILOSOPHY.md` for the design verdict, principles, and contract.
Public surface: one primitive (`render`), one Protocol describing the renderer
signature (`RenderFn`), and one exception family (`RenderError` +
`ShapeMismatch` + `DTypeMismatch`).
"""

from loom.errors import DTypeMismatch, RenderError, ShapeMismatch
from loom.render import RenderFn, render


__all__ = [
    "DTypeMismatch",
    "RenderError",
    "RenderFn",
    "ShapeMismatch",
    "render",
]
