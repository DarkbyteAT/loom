"""Run each examples/0*.py as a subprocess and assert it exits cleanly.

The examples are scripts, not importable modules, so we exec them as
subprocesses and check returncode + stdout. Missing files skip gracefully so
this suite can run before the examples-A/B PRs merge.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = REPO_ROOT / "examples"
TIMEOUT_S = 30

EXPECTED = [
    ("01", "one_inr_per_weight"),
    ("02", "shared_functa_with_film"),
    ("03", "hypernet_vmap"),
    ("04", "hypernet_scan_contexts"),
    ("05", "inner_loop_adaptation"),
    ("06", "heterogeneous_per_leaf"),
]


def _find_example(prefix: str) -> Path | None:
    if not EXAMPLES_DIR.is_dir():
        return None
    matches = sorted(EXAMPLES_DIR.glob(f"{prefix}_*.py"))
    return matches[0] if matches else None


@pytest.mark.integration
@pytest.mark.parametrize(("prefix", "slug"), EXPECTED, ids=[p for p, _ in EXPECTED])
def test_example_runs(prefix: str, slug: str) -> None:
    # Given: an example script for this pattern
    script = _find_example(prefix)
    if script is None:
        pytest.skip(f"examples/{prefix}_*.py not present yet")

    # When: we run it as a subprocess
    result = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_S,
        cwd=REPO_ROOT,
    )

    # Then: it exits cleanly and prints a PASS sentinel
    assert result.returncode == 0, (
        f"{script.name} exited with {result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "PASS" in result.stdout, f"{script.name} did not print PASS sentinel\nstdout:\n{result.stdout}"
