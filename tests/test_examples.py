"""Run each examples/0*.py as a subprocess and assert it exits cleanly.

The examples are scripts, not importable modules, so we exec them as
subprocesses and check returncode + stdout. Missing files skip gracefully so
this suite can run before the examples-A/B PRs merge.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
# Sized for JAX cold-compile on CI runners; tighter values flake on first-run
# tracing for examples 03 (hypernet vmap) and 05 (inner-loop scan-grad).
TIMEOUT_S = 60

EXPECTED = [
    ("01", "one_inr_per_weight"),
    ("02", "shared_functa_film"),
    ("03", "hypernet_distribution"),
    ("04", "hypernet_conditioning"),
    ("05", "inner_loop_adaptation"),
    ("06", "heterogeneous_f"),
]


@pytest.mark.integration
@pytest.mark.parametrize(("prefix", "slug"), EXPECTED, ids=[f"{p}_{s}" for p, s in EXPECTED])
def test_example_runs(prefix: str, slug: str) -> None:
    # Given: an example script for this pattern
    script = EXAMPLES_DIR / f"{prefix}_{slug}.py"
    # TODO: remove the skip-on-missing branch once the tier-2 wave merges into
    # feat/v01-render-substrate. Before merge it lets this PR's CI run green
    # on a branch that doesn't yet contain the examples; after merge a missing
    # file is a real test failure (silent skip would hide an accidental
    # rename/delete), so this branch should hard-fail instead.
    if not script.is_file():
        pytest.skip(f"examples/{prefix}_{slug}.py not present yet")

    # When: we run it as a subprocess. Python prepends the script's directory
    # (examples/) to sys.path, not the repo root, so we explicitly inject the
    # repo root via PYTHONPATH. Without this, `import loom` from the script
    # depends on loom being installed in the active environment and could
    # silently pick up a stale globally-installed version.
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)

    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_S,
            cwd=REPO_ROOT,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        # capture_output=True swallows the partial stdout/stderr into the
        # exception; surface them so CI failures are debuggable.
        pytest.fail(f"{script.name} timed out after {TIMEOUT_S}s\nstdout:\n{exc.stdout!r}\nstderr:\n{exc.stderr!r}")

    # Then: it exits cleanly and prints a PASS sentinel
    assert result.returncode == 0, (
        f"{script.name} exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PASS" in result.stdout, f"{script.name} did not print PASS sentinel\nstdout:\n{result.stdout}"
