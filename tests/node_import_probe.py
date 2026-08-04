"""Import every pinned custom node inside the image.

A node whose dependencies did not make it into the lock fails at workflow-load
time -- which means it fails in front of you, mid-session, not in CI. Importing
each one at build-verification time moves that failure to where it belongs.
"""

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, "/opt/comfyui")

# ComfyUI parses sys.argv at import of comfy.cli_args, and importing almost any
# custom node reaches comfy.model_management, which selects a device eagerly and
# raises "Found no NVIDIA driver on your system" on a GPU-less machine. Ask for
# CPU before anything comfy-related loads, or this probe only ever passes on a
# host that has the very hardware it is meant to not need.
sys.argv = ["main.py", "--cpu"]

lock = json.loads(Path("/opt/comfyui/nodes.lock.json").read_text())
failures = []

for node in lock["nodes"]:
    directory = Path("/opt/comfyui/custom_nodes") / node["name"]
    init = directory / "__init__.py"
    if not init.exists():
        failures.append(f"{node['name']}: no __init__.py at pinned commit {node['commit'][:12]}")
        continue
    try:
        name = "cn_" + node["name"].replace("-", "_").replace(".", "_")
        spec = importlib.util.spec_from_file_location(
            name, init, submodule_search_locations=[str(directory)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 -- any import failure is a failure
        failures.append(f"{node['name']}: {type(exc).__name__}: {exc}")

for failure in failures:
    print(f"IMPORT-FAIL {failure}", file=sys.stderr)

print("ALL-IMPORT-OK" if not failures else "IMPORT-FAILURES")
sys.exit(1 if failures else 0)
