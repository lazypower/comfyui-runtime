# ComfyUI runtime

One definition, two pinned images, published to
`ghcr.io/lazypower/comfyui:main-cuda` and `ghcr.io/lazypower/comfyui:main-rocm`.

Deterministic runtime, externalized durable state. The image is a complete,
immutable description of the running system; everything that accumulates bytes
lives on the host mount.

## Layout

```
manifest/comfyui.toml       declared intent -- edit this
manifest/nodes.lock.json    resolved commits -- generated, committed
env/<backend>/requirements.txt   resolved wheels + hashes -- generated, committed
containers/Containerfile    shared by both backends
```

`comfyui.toml` is never read at build time. The build consumes only the two
generated lockfiles, so an image is reproducible from the lock alone.

## Backends

Neither image uses a vendor base image. Torch's wheels bundle their own
accelerator userspace (`nvidia-*` / HIP + rocBLAS + MIOpen); the kernel driver
comes from the host. Both build from `ubuntu:24.04`, and the entire backend
delta is one wheel index plus a few apt packages.

|        | index                    | torch          | host wiring |
|--------|--------------------------|----------------|-------------|
| `cuda` | `.../whl/cu130`          | `2.13.0+cu130` | CDI: `--device nvidia.com/gpu=all` |
| `rocm` | `.../whl/rocm7.1`        | `2.13.0+rocm7.1` | `--device /dev/kfd --device /dev/dri --group-add keep-groups` |

Same torch version on both, deliberately — custom nodes then behave identically
across hosts. ROCm 7.1 is what buys that parity (6.4 tops out at torch 2.9.1).
If the AMD host's `amdgpu` kernel driver is too old for ROCm 7, change the one
`index` line in `comfyui.toml` back to `rocm6.4` and re-lock.

The backend owns `torch`/`torchvision`/`torchaudio`, pinned to their *local*
version (`+cu130`, `+rocm7.1`). Node-supplied pins for those are stripped during
collection. A bare `torch==` resolves to the generic CUDA-linked PyPI wheel even
with a backend index attached — which produces a ROCm image that imports cleanly
and then finds no GPU.

## Durable state

Bind-mount `/var/mnt/diffusion` at the same path inside and outside the
container, so paths in workflows and logs mean the same thing in both places.

```
/var/mnt/diffusion/
  models/            model tree (extra_model_paths.yaml, is_default)
  input/  output/    --input-directory / --output-directory
  user/              workflows and settings
  cache/             HF_HOME, TORCH_HOME, XDG_CACHE_HOME, temp
  custom_nodes.dev/  development scratch -- see below
```

`just init-state` creates the tree. The entrypoint refuses to start if the mount
is missing, rather than writing a model tree into the container's upper layer.
For the same reason the Containerfile deliberately declares no `VOLUME`: a
declared volume would be auto-created on an unmounted run and satisfy that check.

### SELinux hosts (Bazzite, CoreOS)

Run `just label-state` **once** before the first container start:

```
semanage fcontext -a -t container_file_t '/var/mnt/diffusion(/.*)?'
restorecon -RF /var/mnt/diffusion
```

Deliberately not a `:z` mount flag. `:z` relabels the entire tree on every
container start — across a multi-terabyte model directory that is minutes to
hours of pointless I/O, every single time. The persistent label survives reboots
and relabels, and the quadlet then needs no mount flag at all.

Bazzite ships no `semanage` by default:
`rpm-ostree install policycoreutils-python-utils`.

## Custom nodes

Production nodes are declared in `comfyui.toml`, pinned to exact commits, with
their dependencies resolved into the same lock as everything else.

```
edit manifest/comfyui.toml   ->   just lock   ->   just build   ->   commit lockfiles
```

`/var/mnt/diffusion/custom_nodes.dev` is mounted and on the node search path, so
a dependency-free node can be dropped in and tried live. The production venv is
owned by root and the service account is not root, so a node needing Python
dependencies **will** fail to import there. That failure is the promotion signal:
add it to the manifest and rebuild.

ComfyUI-Manager ships for discovery and missing-node inspection via `cm-cli`
(`just inspect <workflow.json>`). It is not the dependency authority and cannot
become one — the read-only venv enforces that, not configuration.

Consequence worth knowing: nodes that persist settings into their own directory
rather than the user directory will log write errors. That is the immutability
working, not a defect.

## Operating

```
just lock                    resolve intent -> commits -> wheels
just lock-check              fail if the committed lockfiles are stale
just build                   both images
just build-one cuda          one
just test                    manifest checks + image contract (no GPU needed)
just test-gpu rocm           accelerator reachable -- run ON the target host
just run cuda                local run against $COMFY_STATE
just inspect wf.json         what does Manager think is missing
just manifest-of cuda        the pinned set inside a built image
```

A host can build the backend it runs rather than pulling it: `just build-one
cuda`, then `just test-gpu cuda` against the real card. The images are 7–13GB
and need roughly twice that transiently while the layer is committed, so a
builder wants ~40GB free.

### What is provable where

CI runs on GitHub-hosted runners, which have **no GPU**. That bounds what it can
claim, but less than it sounds — the common failure is a node whose dependency
never made it into the lock, and that is fully provable on CPU.

CI proves: the dependency closure resolves and hash-verifies, every pinned node
imports, the venv is immutable, the state mount is enforced, ComfyUI registers
its nodes and serves on CPU, and **the torch wheel is the correct accelerator
variant** — `torch.version.hip` is a property of the wheel, not the machine, so
a ROCm image that silently shipped the generic CUDA wheel fails here.

CI cannot prove that kernels actually run. `just test-gpu <backend>` covers that
and must run on the target host. It allocates on-device and does a real matmul,
because `torch.cuda.is_available()` has returned true on hosts that could not
then execute anything.

`just test-gpu <backend>` is a post-deploy step until a runner with the
matching GPU exists.

## Deployment

This repo ships images, tests, and the justfile; unit files live with the rest
of your fleet configuration. A unit needs to get four things right — the two
backends differ only in the first:

- **Device access.** Passing `--device` is not sufficient on its own: `/dev/kfd`
  and `/dev/dri/renderD128` are owned by `render`/`video`. The image creates
  those groups at Fedora's GIDs (105 / 39, both build args) and puts the service
  account in them. Rootless podman can instead use `--group-add keep-groups`.
- **Shared memory.** `--shm-size=8g`. Torch dataloader workers communicate over
  `/dev/shm`, and the 64MB default stalls silently under load.
- **uid.** The service account is uid 1000 (`COMFY_UID` build arg) and must match
  the owner of the state tree.
- **No `:z` on the mount** — see the SELinux note above.

The host does not need ROCm or CUDA installed. Accelerator userspace rides
inside the torch wheels; the host supplies only the kernel driver and the
device nodes. Recent Bazzite images
[no longer ship ROCm](https://lunar.computer/bazzite-removes-qemu-and-rocm-from-base-images-20260324)
and that is not an obstacle.
