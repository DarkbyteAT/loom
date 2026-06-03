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
# Opt-in escape hatch: set this env var to skip-on-missing instead of failing
# loudly. Used during the pre-merge migration window when example files live
# on sibling branches; after the wave merges the env var (and the surrounding
# branch in test_example_runs) become dead and can be removed.
ALLOW_MISSING = "LOOM_ALLOW_MISSING_EXAMPLES"

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
    # TODO: remove this whole branch once the tier-2 wave merges into
    # feat/v01-render-substrate. After merge, a missing example is a real bug
    # (rename/delete drift) and should fail loudly, not skip silently. Until
    # then, opt-in via LOOM_ALLOW_MISSING_EXAMPLES=1 to let cross-branch CI
    # runs (this PR before merge) stay green.
    if not script.is_file():
        if os.environ.get(ALLOW_MISSING):
            pytest.skip(f"examples/{prefix}_{slug}.py not present yet")
        pytest.fail(
            f"examples/{prefix}_{slug}.py is missing. Set {ALLOW_MISSING}=1 "
            "to opt into skip-on-missing during cross-branch CI runs."
        )

    # When: we run it as a subprocess. Python prepends the script's directory
    # (examples/) to sys.path, not the repo root, so we explicitly inject the
    # repo root via PYTHONPATH. Without this, `import loom` from the script
    # depends on loom being installed in the active environment and could
    # silently pick up a stale globally-installed version.
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{REPO_ROOT}{os.pathsep}{existing}" if existing else str(REPO_ROOT)
    # Pin JAX to CPU: these are smoke tests, GPU/TPU yields no benefit, and CI
    # runners can OOM or contend on accelerators. JAX_PLATFORMS is the current
    # spelling (the older JAX_PLATFORM_NAME still works but is deprecated).
    env["JAX_PLATFORMS"] = "cpu"

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
        # exception; surface them so CI failures are debuggable. Use bare
        # interpolation (no !r) so JAX tracebacks stay multi-line and readable
        # — same formatting as the assert-failure path below.
        pytest.fail(f"{script.name} timed out after {TIMEOUT_S}s\nstdout:\n{exc.output}\nstderr:\n{exc.stderr}")

    # Then: it exits cleanly and prints a PASS sentinel
    assert result.returncode == 0, (
        f"{script.name} exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PASS" in result.stdout, f"{script.name} did not print PASS sentinel\nstdout:\n{result.stdout}"
