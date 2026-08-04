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

# Importing almost any custom node reaches comfy.model_management, which selects
# a device eagerly and raises on a GPU-less machine ("Found no NVIDIA driver" /
# "No CUDA GPUs are available"). So ask for CPU before anything comfy loads.
#
# Setting sys.argv is NOT enough on its own. comfy/options.py defaults
# args_parsing=False, and comfy/cli_args.py then does `parser.parse_args([])` --
# discarding sys.argv completely. main.py calls enable_args_parsing() before it
# imports cli_args; anything else driving ComfyUI has to do the same, or --cpu
# is accepted in silence and ignored.
STATE = "/var/mnt/diffusion"
sys.argv = [
    "main.py",
    "--cpu",
    # Point every writable directory at the state mount, exactly as the
    # entrypoint does. Without this the managers constructed below try to
    # populate /opt/comfyui/user, which is deliberately read-only.
    "--user-directory", f"{STATE}/user",
    "--input-directory", f"{STATE}/input",
    "--output-directory", f"{STATE}/output",
    "--temp-directory", f"{STATE}/cache/temp",
]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from comfy.cli_args import args  # noqa: E402

assert args.cpu, "enable_args_parsing() did not take effect; --cpu was discarded"

# Parsing the directory arguments does not move anything on its own -- main.py
# pushes each one into folder_paths by hand (main.py:135-156, :489-492). Skip
# this and UserManager below happily tries to create /opt/comfyui/user, which
# is read-only by design.
import os  # noqa: E402

import folder_paths  # noqa: E402

folder_paths.set_output_directory(os.path.abspath(args.output_directory))
folder_paths.set_input_directory(os.path.abspath(args.input_directory))
folder_paths.set_user_directory(os.path.abspath(args.user_directory))
folder_paths.set_temp_directory(os.path.join(os.path.abspath(args.temp_directory), "temp"))

# Nodes reach for server.PromptServer.instance at import time (rgthree,
# VideoHelperSuite, KJNodes' LTXV nodes all do). main.py constructs it before
# loading custom nodes; constructing it here is what makes this probe resemble
# a real startup rather than testing an arrangement that never occurs.
import asyncio  # noqa: E402

import server  # noqa: E402

asyncio.set_event_loop(loop := asyncio.new_event_loop())
server.PromptServer(loop)

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
