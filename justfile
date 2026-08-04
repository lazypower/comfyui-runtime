# ComfyUI images -- one definition, two pinned backends.

engine   := env("CONTAINER_ENGINE", "podman")
registry := env("REGISTRY", "ghcr.io/lazypower")
platform := "linux/amd64"
state    := env("COMFY_STATE", "/var/mnt/diffusion")

# Extra args for `run`. Firecracker CI needs --net=host (no nftables in that
# kernel, so netavark cannot build a nested netns).
run_args := env("RUN_ARGS", "")

# Torch dataloader workers communicate over /dev/shm; the 64MB default is a
# classic silent stall under load.
shm := "8g"

_default:
    @just --list --unsorted

# ---------------------------------------------------------------- locking

# Resolve intent -> exact commits -> exact wheels. Commit the results.
lock: resolve && (compile "cuda") (compile "rocm")

# Resolve every ref in comfyui.toml to an exact commit.
resolve:
    python3 scripts/pin.py resolve

# Compile one backend's fully-pinned, hash-verified requirements.
compile backend:
    python3 scripts/pin.py collect {{backend}}
    uv pip compile env/{{backend}}/requirements.in \
        --output-file env/{{backend}}/requirements.txt \
        --extra-index-url "$(python3 -c "import json;print(json.load(open('manifest/nodes.lock.json'))['backends']['{{backend}}']['index'])")" \
        --index-strategy unsafe-best-match \
        --python-version "$(python3 -c "import json;print(json.load(open('manifest/nodes.lock.json'))['runtime']['python'])")" \
        --python-platform "$(python3 -c "import json;print(json.load(open('manifest/nodes.lock.json'))['runtime']['platform'])")" \
        --generate-hashes \
        --emit-index-url \
        --no-annotate \
        --custom-compile-command "just compile {{backend}}"

# CI gate: do the committed lockfiles still agree with the committed pins?
# Shared with CI -- see scripts/lock_check.py for why it does not re-resolve.
lock-check:
    python3 scripts/lock_check.py

# ---------------------------------------------------------------- building

# Build both images.
build: (build-one "cuda") (build-one "rocm")

# Build one backend image.
build-one backend tag="dev":
    {{engine}} build \
        --platform {{platform}} \
        --file containers/Containerfile \
        --build-arg BACKEND={{backend}} \
        --build-arg BACKEND_APT="$(python3 -c "import json;print(' '.join(json.load(open('manifest/nodes.lock.json'))['backends']['{{backend}}']['apt']))")" \
        --build-arg PYTHON_VERSION="$(python3 -c "import json;print(json.load(open('manifest/nodes.lock.json'))['runtime']['python'])")" \
        --tag {{registry}}/comfyui:{{tag}}-{{backend}} \
        .

push tag="dev": (push-one "cuda" tag) (push-one "rocm" tag)

push-one backend tag="dev":
    {{engine}} push {{registry}}/comfyui:{{tag}}-{{backend}}

# ---------------------------------------------------------------- testing

# Everything that does not need a GPU.
test: test-manifest test-image

# Static checks against the manifest and lockfiles. No container required.
test-manifest:
    uv run --with pytest python -m pytest tests/test_manifest.py -q

# Boot each image and assert the runtime contract. No GPU required.
test-image: (test-image-one "cuda") (test-image-one "rocm")

test-image-one backend tag="dev":
    ENGINE={{engine}} IMAGE={{registry}}/comfyui:{{tag}}-{{backend}} BACKEND={{backend}} \
        RUN_ARGS="{{run_args}}" bash tests/smoke.sh

# Assert the accelerator is actually visible. Run this ON the target host.
test-gpu backend tag="dev":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{backend}}" in
      cuda) dev=(--device nvidia.com/gpu=all) ;;
      rocm) dev=(--device /dev/kfd --device /dev/dri --group-add keep-groups) ;;
      *) echo "unknown backend {{backend}}" >&2; exit 1 ;;
    esac
    {{engine}} run --rm -i "${dev[@]}" --entrypoint python \
        {{registry}}/comfyui:{{tag}}-{{backend}} - < tests/gpu_probe.py

# ---------------------------------------------------------------- running

# Run locally against a state tree. Defaults to $COMFY_STATE.
run backend="cuda" tag="dev":
    #!/usr/bin/env bash
    set -euo pipefail
    case "{{backend}}" in
      cuda) dev=(--device nvidia.com/gpu=all) ;;
      rocm) dev=(--device /dev/kfd --device /dev/dri --group-add keep-groups) ;;
    esac
    {{engine}} run --rm -it {{run_args}} \
        "${dev[@]}" \
        --shm-size {{shm}} \
        --publish 8188:8188 \
        --volume {{state}}:/var/mnt/diffusion \
        {{registry}}/comfyui:{{tag}}-{{backend}}

# Create the durable state tree at {{state}}.
init-state:
    #!/usr/bin/env bash
    set -euo pipefail
    for d in models/{checkpoints,clip,clip_vision,configs,controlnet,diffusers,diffusion_models,embeddings,gligen,hypernetworks,loras,photomaker,style_models,text_encoders,upscale_models,vae,vae_approx} \
             input output user cache custom_nodes.dev; do
        mkdir -p "{{state}}/$d"
    done
    echo "state tree ready at {{state}}"
    if command -v getenforce >/dev/null && [ "$(getenforce)" != Disabled ]; then
        echo "SELinux is $(getenforce) -- run 'just label-state' before first container start"
    fi

# Label durable state for container access on an SELinux host (Bazzite, CoreOS).
#
# Done ONCE, persistently, rather than with a :z mount flag. :z relabels the
# entire tree on every container start -- across a multi-terabyte model
# directory that is minutes to hours of pointless I/O, every time. This survives
# reboots and relabels, and the quadlet then needs no mount flag at all.
label-state:
    #!/usr/bin/env bash
    set -euo pipefail
    if ! command -v semanage >/dev/null; then
        echo "semanage not found; install policycoreutils-python-utils" >&2
        echo "  rpm-ostree install policycoreutils-python-utils   (reboot required)" >&2
        exit 1
    fi
    # SELinux ships an equivalency rule making /var/mnt an alias of /mnt, and
    # fcontext specs must name the canonical path -- semanage rejects the alias
    # outright ("conflicts with equivalency rule '/var/mnt /mnt'"). restorecon
    # still takes the real path; only the spec is rewritten.
    spec='{{state}}'
    case "$spec" in
        /var/mnt/*) spec="${spec#/var}" ;;
    esac
    sudo semanage fcontext -a -t container_file_t "${spec}(/.*)?" 2>/dev/null \
      || sudo semanage fcontext -m -t container_file_t "${spec}(/.*)?"
    sudo restorecon -RvF {{state}}
    echo "{{state}} labelled container_file_t via ${spec} (persistent)"

# What did Manager find missing for a given workflow? Inspection only.
inspect workflow backend="cuda" tag="dev":
    {{engine}} run --rm {{run_args}} \
        --volume {{state}}:/var/mnt/diffusion \
        --volume "{{workflow}}":/tmp/workflow.json:ro \
        --entrypoint python \
        {{registry}}/comfyui:{{tag}}-{{backend}} \
        custom_nodes/ComfyUI-Manager/cm-cli.py show missing-nodes /tmp/workflow.json

# Print the pinned set for a built image.
manifest-of backend="cuda" tag="dev":
    {{engine}} run --rm --entrypoint cat {{registry}}/comfyui:{{tag}}-{{backend}} /opt/comfyui/nodes.lock.json
