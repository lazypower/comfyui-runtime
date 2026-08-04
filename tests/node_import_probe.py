"""Load the pinned custom nodes the way ComfyUI itself does, and report which
ones failed.

A node whose dependencies did not make it into the lock fails at workflow-load
time -- mid-session, in front of you, not in CI. Loading each one at
build-verification time moves that failure to where it belongs.

This drives ComfyUI's own loader rather than importing node packages by hand.
Hand-importing reproduces neither the order nor the state of a real startup:
builtin extras load first (importing custom nodes before them yields spurious
circular-import errors), args must be applied to folder_paths rather than
merely parsed, and PromptServer.instance must exist. Every one of those was a
false failure here before this probe used init_extra_nodes.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

COMFY = "/opt/comfyui"
STATE = "/var/mnt/diffusion"

sys.path.insert(0, COMFY)

# ComfyUI reads sys.argv when comfy.cli_args is imported -- but only if arg
# parsing was explicitly enabled. comfy/options.py defaults args_parsing=False,
# and cli_args.py then calls parse_args([]), discarding sys.argv entirely.
sys.argv = [
    "main.py",
    "--cpu",
    "--user-directory", f"{STATE}/user",
    "--input-directory", f"{STATE}/input",
    "--output-directory", f"{STATE}/output",
    "--temp-directory", f"{STATE}/cache/temp",
    "--disable-api-nodes",  # keeps this probe off the network
]

import comfy.options  # noqa: E402

comfy.options.enable_args_parsing()

from comfy.cli_args import args  # noqa: E402

assert args.cpu, "enable_args_parsing() did not take effect; --cpu was discarded"

# Parsing moves nothing on its own; main.py pushes each directory into
# folder_paths by hand (main.py:135-156, :489-492). Without this, UserManager
# below creates /opt/comfyui/user instead of the state mount.
import folder_paths  # noqa: E402

folder_paths.set_output_directory(os.path.abspath(args.output_directory))
folder_paths.set_input_directory(os.path.abspath(args.input_directory))
folder_paths.set_user_directory(os.path.abspath(args.user_directory))
folder_paths.set_temp_directory(os.path.join(os.path.abspath(args.temp_directory), "temp"))

# Nodes reach for server.PromptServer.instance at import (rgthree,
# VideoHelperSuite, KJNodes' LTXV nodes). main.py constructs it before loading
# custom nodes.
import server  # noqa: E402

asyncio.set_event_loop(loop := asyncio.new_event_loop())
server.PromptServer(loop)

# The real loader: builtin extras first, then custom_nodes. Same call main.py
# makes at main.py:504.
import nodes  # noqa: E402

loop.run_until_complete(nodes.init_extra_nodes(init_custom_nodes=True, init_api_nodes=False))

loaded = {os.path.realpath(p) for p in nodes.LOADED_MODULE_DIRS.values()}
lock = json.loads(Path(f"{COMFY}/nodes.lock.json").read_text())

failures = [
    node["name"]
    for node in lock["nodes"]
    if os.path.realpath(f"{COMFY}/custom_nodes/{node['name']}") not in loaded
]

for name in failures:
    print(f"IMPORT-FAIL {name} did not register (see ComfyUI's log above)", file=sys.stderr)

print(f"loaded {len(lock['nodes']) - len(failures)}/{len(lock['nodes'])} pinned nodes; "
      f"{len(nodes.NODE_CLASS_MAPPINGS)} node classes registered")
print("ALL-IMPORT-OK" if not failures else "IMPORT-FAILURES")
sys.exit(1 if failures else 0)
