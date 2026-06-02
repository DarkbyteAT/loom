"""Exception family for loom's render contract.

The hierarchy mirrors the guarantees in PHILOSOPHY.md:

```
RenderError                 # base; wraps any exception raised by f
├── ShapeMismatch           # Guarantee 2 violation
└── DTypeMismatch           # Guarantee 3 violation
```

Every error carries the leaf's `path` (a `jax.tree_util.KeyPath`), the expected
`shape` and `dtype`, and — for the subclasses — what `f` actually returned. The
`__str__` is path-pointing so users can locate the offending leaf without
re-reading their pytree.
"""

from __future__ import annotations

from typing import Any


KeyPath = tuple[Any, ...]


def _format_path(path: KeyPath) -> str:
    """Render a `KeyPath` for display.

    JAX's key-part types all implement a useful `__str__`, so we delegate to
    them and join with the empty string — `(GetAttrKey('a'), SequenceKey(0))`
    becomes `.a[0]`, which is the same convention `jax.tree_util` uses for
    `keystr`.
    """
    if not path:
        return "<root>"
    from jax.tree_util import keystr  # pyright: ignore[reportUnknownVariableType]

    return str(keystr(path))


class RenderError(Exception):
    """Base class for failures during `loom.render`.

    Raised when the user-supplied `f` itself raises an exception. The original
    exception is chained via `raise ... from e` so `__cause__` is preserved.

    Attributes:
        path: `KeyPath` of the leaf being rendered when the error occurred.
        shape: Expected shape of the leaf (the shape `f` was asked to produce).
        dtype: Expected dtype of the leaf.
    """

    def __init__(self, path: KeyPath, shape: tuple[int, ...], dtype: Any) -> None:
        """Store leaf context and build the message via `_message`."""
        self.path = path
        self.shape = shape
        self.dtype = dtype
        super().__init__(self._message())

    def _message(self) -> str:
        return f"render failed at leaf {_format_path(self.path)} (expected shape={self.shape}, dtype={self.dtype})"


class ShapeMismatch(RenderError):
    """`f` returned an array whose shape disagrees with the leaf's shape.

    Violation of Guarantee 2 (shape correctness). Loom does not broadcast or
    reshape — the user's `f` is responsible for producing a tensor of exactly
    `leaf.shape`.

    Attributes:
        actual_shape: Shape `f` actually returned.
    """

    def __init__(
        self,
        path: KeyPath,
        expected_shape: tuple[int, ...],
        actual_shape: tuple[int, ...],
        dtype: Any,
    ) -> None:
        """Store the offending actual shape and delegate to `RenderError`."""
        self.actual_shape = actual_shape
        super().__init__(path, expected_shape, dtype)

    def _message(self) -> str:
        return (
            f"shape mismatch at leaf {_format_path(self.path)}: "
            f"f returned shape={self.actual_shape}, expected shape={self.shape}"
        )


class DTypeMismatch(RenderError):
    """`f` returned an array whose dtype disagrees with the leaf's dtype.

    Violation of Guarantee 3 (dtype rule). Loom validates rather than casts —
    silent promotion (e.g. fp32 → fp64) is a footgun the substrate refuses to
    paper over. Users who want a conversion do it inside `f`.

    Attributes:
        actual_dtype: Dtype `f` actually returned.
    """

    def __init__(
        self,
        path: KeyPath,
        shape: tuple[int, ...],
        expected_dtype: Any,
        actual_dtype: Any,
    ) -> None:
        """Store the offending actual dtype and delegate to `RenderError`."""
        self.actual_dtype = actual_dtype
        super().__init__(path, shape, expected_dtype)

    def _message(self) -> str:
        return (
            f"dtype mismatch at leaf {_format_path(self.path)}: "
            f"f returned dtype={self.actual_dtype}, expected dtype={self.dtype}"
        )
