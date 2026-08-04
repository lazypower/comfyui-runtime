#!/usr/bin/env bash
# Runtime contract for a built image. No GPU required -- ComfyUI runs --cpu here.
#
#   ENGINE=docker IMAGE=... BACKEND=cuda bash tests/smoke.sh
set -uo pipefail

ENGINE="${ENGINE:-docker}"
IMAGE="${IMAGE:?set IMAGE}"
BACKEND="${BACKEND:-cuda}"

STATE="$(mktemp -d)"
trap 'rm -rf "$STATE"' EXIT
mkdir -p "$STATE"/{models,input,output,user,cache/temp,custom_nodes.dev}

# A real, dependency-free node in the dev mount. Asserting it reaches
# /object_info proves the promotion workflow's first half end to end -- drop a
# node in, try it live, no rebuild -- which grepping the log for a path name
# never did.
mkdir -p "$STATE/custom_nodes.dev/smoke_probe_node"
cat > "$STATE/custom_nodes.dev/smoke_probe_node/__init__.py" <<'NODE'
class SmokeProbeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}
    RETURN_TYPES = ()
    FUNCTION = "run"
    CATEGORY = "smoke"

    def run(self):
        return ()

NODE_CLASS_MAPPINGS = {"SmokeProbeNode": SmokeProbeNode}
NODE_DISPLAY_NAME_MAPPINGS = {"SmokeProbeNode": "Smoke Probe"}
NODE

# mktemp -d gives 0700 owned by whoever runs this, and the container is uid 1000
# regardless. On a CI runner those differ, so the service account cannot even
# traverse into the mount and every state-dependent check fails looking like a
# missing directory. A throwaway fixture can be world-writable; a real
# deployment matches uid instead (COMFY_UID -- see README).
chmod -R 0777 "$STATE"

pass=0 fail=0
ok()   { printf '  \033[32mok\033[0m   %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$1"; fail=$((fail+1)); }
check() { if [ "$1" = 0 ]; then ok "$2"; else bad "$2"; fi; }

# RUN_ARGS is intentionally unquoted: Firecracker CI passes --net=host through it.
# shellcheck disable=SC2086
mounted() { "$ENGINE" run --rm ${RUN_ARGS:-} --volume "$STATE:/var/mnt/diffusion" "$@"; }

echo "smoke: $IMAGE ($BACKEND)"

# --- durable state is mandatory ---------------------------------------------
# RUN_ARGS matters even here: without --net=host on a Firecracker runner the
# container never starts, and this would "fail" for the wrong reason.
# shellcheck disable=SC2086
if out=$("$ENGINE" run --rm ${RUN_ARGS:-} --entrypoint /usr/local/bin/entrypoint "$IMAGE" --cpu 2>&1); then
    bad "started without durable state mounted (would write into the image layer)"
elif grep -q "not a mount point" <<<"$out"; then
    ok "refuses to start without durable state mounted"
else
    bad "failed without a mount, but not for the expected reason: $(head -3 <<<"$out")"
fi

# --- the production environment is immutable --------------------------------
mounted --entrypoint bash "$IMAGE" -c 'test ! -w /opt/comfyui/.venv' 2>/dev/null
check $? "production venv is not writable by the service account"

mounted --entrypoint bash "$IMAGE" -c \
    'pip install --quiet six 2>/dev/null && exit 1 || exit 0' 2>/dev/null
check $? "runtime pip install into the production venv fails"

mounted --entrypoint bash "$IMAGE" -c 'test "$(id -u)" -ne 0' 2>/dev/null
check $? "runs as a non-root service account"

# The other half of the contract: the app tree IS writable, because ComfyUI's
# ecosystem writes into it at import time -- Manager into its own directory,
# Custom-Scripts into web/extensions. Those writes land in the container's
# upper layer and vanish on restart, so drift cannot accumulate. The venv
# checked above is the line that matters.
mounted --entrypoint bash "$IMAGE" -c 'test -w /opt/comfyui/custom_nodes' 2>/dev/null
check $? "custom_nodes is writable (node self-config at import)"

# Exactly what ComfyUI-Custom-Scripts does at import (pysssss.py:
# os.makedirs(get_comfy_dir("web/extensions/pysssss"))). Asserting the
# capability rather than a path's existence: ComfyUI 0.27 ships no web/
# directory at all -- the front end arrives via comfyui-frontend-package -- so
# `test -w /opt/comfyui/web` would fail on a tree that works perfectly.
mounted --entrypoint bash "$IMAGE" -c \
    'mkdir -p /opt/comfyui/web/extensions/_probe' 2>/dev/null
check $? "nodes can create front-end extension dirs in the app tree"

# --- the pinned environment is what we said it was --------------------------
mounted --entrypoint python "$IMAGE" -c 'import torch; print(torch.__version__)' >/dev/null 2>&1
check $? "torch imports"

# Which accelerator the wheel was BUILT for is a property of the wheel, not the
# machine -- so this holds on a GPU-less CI runner. It is also the check that
# catches the silent killer: a ROCm image that quietly shipped the generic
# CUDA-linked PyPI wheel, imports cleanly, and finds no GPU on the AMD host.
probe=$(mounted --entrypoint python "$IMAGE" -c \
    'import torch; print(torch.__version__, torch.version.cuda, torch.version.hip)' 2>/dev/null)
read -r ver cuda_ver hip_ver <<<"$probe"
case "$BACKEND" in
    cuda) if [ "$hip_ver" = None ] && [ "$cuda_ver" != None ]; then
              ok "torch is a CUDA build (${ver}, cuda ${cuda_ver})"
          else bad "expected a CUDA build, got version=$ver cuda=$cuda_ver hip=$hip_ver"; fi ;;
    rocm) if [ "$hip_ver" != None ] && [ -n "$hip_ver" ]; then
              ok "torch is a ROCm build (${ver}, hip ${hip_ver})"
          else bad "expected a ROCm build, got version=$ver cuda=$cuda_ver hip=$hip_ver"; fi ;;
esac

# every pinned node must import -- a node that only fails at workflow-load time
# is a node that fails in front of you, not in CI
mounted -i --entrypoint python "$IMAGE" - < "$(dirname "$0")/node_import_probe.py"
check $? "every pinned custom node imports"

# --- nothing accumulates inside the image -----------------------------------
mounted --entrypoint bash "$IMAGE" -c \
    '[ "$HF_HOME" = /var/mnt/diffusion/cache/huggingface ]' 2>/dev/null
check $? "caches are redirected onto durable state"

# --- it actually serves ------------------------------------------------------
# No -P: every probe below goes through `exec`, so nothing needs publishing --
# and `-P` conflicts with the --net=host that Firecracker requires.
cid=$(mounted -d --entrypoint /usr/local/bin/entrypoint "$IMAGE" \
        --cpu --listen 127.0.0.1 --port 8188 2>/dev/null)
if [ -n "$cid" ]; then
    booted=1
    for _ in $(seq 1 90); do
        if "$ENGINE" exec "$cid" python -c \
            'import urllib.request;urllib.request.urlopen("http://127.0.0.1:8188/system_stats",timeout=2)' \
            >/dev/null 2>&1; then booted=0; break; fi
        sleep 2
    done
    check $booted "ComfyUI serves /system_stats"

    "$ENGINE" exec "$cid" python -c '
import json,urllib.request
o=json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info",timeout=20))
import sys; sys.exit(0 if len(o)>100 else 1)' >/dev/null 2>&1
    check $? "node registry populated"

    "$ENGINE" exec "$cid" python -c '
import json, sys, urllib.request
info = json.load(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=20))
sys.exit(0 if "SmokeProbeNode" in info else 1)' >/dev/null 2>&1
    check $? "a node in the dev mount loads without a rebuild"

    # The database default is computed from the source tree, not the user
    # directory, so it silently lands in the ephemeral layer unless overridden.
    if "$ENGINE" logs "$cid" 2>&1 | grep -q "Failed to initialize database"; then
        bad "database initialises"
    else
        ok "database initialises"
    fi

    "$ENGINE" exec "$cid" test -f /var/mnt/diffusion/user/comfyui.db 2>/dev/null
    check $? "database lives on durable state"

    "$ENGINE" rm -f "$cid" >/dev/null 2>&1
else
    bad "container failed to start"
fi

echo
echo "  $pass passed, $fail failed"
[ "$fail" -eq 0 ]
