#!/usr/bin/env bash
# Startup contract:
#   1. durable state is mounted and writable, or we fail loudly
#   2. the production environment is NOT writable, or we fail loudly
#   3. every path that accumulates bytes points at the mount, not the image
set -euo pipefail

COMFY_ROOT="${COMFY_ROOT:-/opt/comfyui}"
STATE="${COMFY_STATE:-/var/mnt/diffusion}"

die() { echo "comfyui: FATAL: $*" >&2; exit 1; }

# --- 1. durable state -------------------------------------------------------
# A bare directory means nothing was bind-mounted. Refuse: writing a model tree
# into the container's upper layer is the failure this whole design exists to
# prevent, and it is silent until the day you restart.
mountpoint -q "$STATE" || [ -n "${COMFY_ALLOW_UNMOUNTED_STATE:-}" ] \
  || die "$STATE is not a mount point. Bind-mount durable state, or set COMFY_ALLOW_UNMOUNTED_STATE=1 for throwaway runs."

for d in models input output user cache custom_nodes.dev; do
  mkdir -p "$STATE/$d" 2>/dev/null || die "cannot create $STATE/$d -- check mount ownership (expected uid $(id -u))"
done
[ -w "$STATE/output" ] || die "$STATE/output is not writable by uid $(id -u)"

# --- 2. the environment is immutable ---------------------------------------
# Enforced by ownership, not politeness: root owns /opt/comfyui, we are not root.
# If this check passes, someone mounted over the venv and the image no longer
# describes the running system.
if [ -w "$COMFY_ROOT/.venv" ]; then
  die "$COMFY_ROOT/.venv is writable -- the production environment must not be mutable. Refusing to start."
fi

# --- 3. nothing accumulates inside the image -------------------------------
export HF_HOME="$STATE/cache/huggingface"
export TORCH_HOME="$STATE/cache/torch"
export XDG_CACHE_HOME="$STATE/cache/xdg"
export MPLCONFIGDIR="$STATE/cache/matplotlib"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
mkdir -p "$HF_HOME" "$TORCH_HOME" "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

# ComfyUI-Manager: discovery and missing-node inspection only. The read-only
# venv is what actually prevents it installing anything; this is the polite
# half of the same statement.
export COMFYUI_MANAGER_NO_AUTO_UPDATE=1
export COMFYUI_MANAGER_NETWORK_MODE=offline
export GIT_TERMINAL_PROMPT=0

echo "comfyui: backend=${COMFY_BACKEND:-unknown} state=$STATE uid=$(id -u)"

# --database-url is NOT covered by --user-directory. comfy/cli_args.py:238
# computes its default at import time as <comfy>/../user/comfyui.db and never
# consults the user directory, so without this the database lands in the
# container's ephemeral layer -- workflow metadata silently discarded on every
# restart, which is precisely what externalising state exists to prevent.
exec python "$COMFY_ROOT/main.py" \
  --base-directory       "$COMFY_ROOT" \
  --extra-model-paths-config "$COMFY_ROOT/extra_model_paths.yaml" \
  --input-directory      "$STATE/input" \
  --output-directory     "$STATE/output" \
  --user-directory       "$STATE/user" \
  --temp-directory       "$STATE/cache/temp" \
  --database-url         "sqlite:///$STATE/user/comfyui.db" \
  "$@"
