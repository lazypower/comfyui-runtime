#!/usr/bin/env python3
"""Do the committed lockfiles still agree with the committed pins?

Deliberately does NOT re-resolve refs. Node refs float (ref = "main"), so
re-resolving would report drift whenever any upstream repo moved -- a gate that
fails for reasons unrelated to the commit under test is a gate people learn to
ignore. `just lock` is how you intentionally pick up upstream movement; this
only asks whether nodes.lock.json and env/<backend>/requirements.in describe the
same world.

Pure stdlib and no subprocesses: CI runners are minimal rootfs images that do
not necessarily ship diffutils.
"""

from __future__ import annotations

import difflib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("cuda", "rocm")

stale = False

for backend in BACKENDS:
    target = ROOT / "env" / backend / "requirements.in"
    committed = target.read_text()

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "pin.py"), "collect", backend],
        check=True,
        capture_output=True,
    )
    regenerated = target.read_text()
    target.write_text(committed)  # leave the tree as we found it

    if committed != regenerated:
        stale = True
        print(f"STALE: env/{backend}/requirements.in disagrees with manifest/nodes.lock.json")
        sys.stdout.writelines(
            difflib.unified_diff(
                committed.splitlines(keepends=True),
                regenerated.splitlines(keepends=True),
                fromfile=f"committed/{backend}/requirements.in",
                tofile=f"regenerated/{backend}/requirements.in",
            )
        )
        print("  run `just lock` and commit the result\n")

if stale:
    sys.exit(1)

print("lockfiles agree with the pinned commits")
