"""Static checks on the manifest and the artifacts it generates.

These run without a container engine and without a GPU. They exist to catch the
class of mistake that is invisible until deploy day: a floating pin, a lockfile
that drifted from its manifest, a backend that quietly resolved onto the wrong
torch build.
"""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKENDS = ("cuda", "rocm")

MANIFEST = tomllib.loads((ROOT / "manifest" / "comfyui.toml").read_text())
LOCK = json.loads((ROOT / "manifest" / "nodes.lock.json").read_text())

SHA40 = re.compile(r"^[0-9a-f]{40}$")


def containerfile_instructions() -> list[str]:
    """The Containerfile with comments stripped and continuations joined.

    Assertions must run against instructions, not prose: this file explains
    itself at length, and a comment quoting `uv pip sync` or `comfyui.toml` is
    not the same as an instruction doing it.
    """
    body = (ROOT / "containers" / "Containerfile").read_text()
    lines = [ln for ln in body.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]

    instructions: list[str] = []
    for line in lines:
        if instructions and instructions[-1].rstrip().endswith("\\"):
            instructions[-1] = instructions[-1].rstrip().removesuffix("\\") + " " + line.strip()
        else:
            instructions.append(line)
    return instructions


# --------------------------------------------------------------- pinning

def test_comfyui_core_is_pinned_to_an_exact_commit():
    assert SHA40.match(LOCK["comfyui"]["commit"])


@pytest.mark.parametrize("node", LOCK["nodes"], ids=lambda n: n["name"])
def test_every_node_is_pinned_to_an_exact_commit(node):
    assert SHA40.match(node["commit"]), f"{node['name']} is not pinned to a commit"


def test_lock_covers_exactly_the_declared_nodes():
    declared = {n["name"] for n in MANIFEST["nodes"]}
    locked = {n["name"] for n in LOCK["nodes"]}
    assert declared == locked, (
        f"manifest and lock disagree; run `just lock`. "
        f"only in manifest: {declared - locked}, only in lock: {locked - declared}"
    )


def test_node_names_are_unique():
    names = [n["name"] for n in MANIFEST["nodes"]]
    assert len(names) == len(set(names))


# --------------------------------------------------------------- backends

@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_has_compiled_requirements(backend):
    assert (ROOT / "env" / backend / "requirements.txt").exists(), (
        f"env/{backend}/requirements.txt missing -- run `just lock`"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_compiled_requirements_are_hash_verified(backend):
    body = (ROOT / "env" / backend / "requirements.txt").read_text()
    assert "--hash=sha256:" in body, (
        f"{backend} lock has no hashes; determinism is not enforced"
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_compiled_requirements_target_the_declared_index(backend):
    body = (ROOT / "env" / backend / "requirements.txt").read_text()
    index = LOCK["backends"][backend]["index"]
    assert index in body, f"{backend} lock does not reference {index}"


def test_backends_resolved_to_different_torch_builds():
    """The whole reason for two locks. If these match, one backend is wrong."""
    def torch_line(backend: str) -> str:
        for line in (ROOT / "env" / backend / "requirements.txt").read_text().splitlines():
            if line.startswith("torch==") or line.startswith("torch @"):
                return line.strip()
        pytest.fail(f"no torch pin found in {backend} lock")

    assert torch_line("cuda") != torch_line("rocm")


@pytest.mark.parametrize("backend", BACKENDS)
def test_nodes_do_not_pin_the_torch_triple(backend):
    """The backend owns torch. A node pin leaking through would silently
    override the accelerator build for the whole image.

    The generated preamble legitimately pins the triple; everything after the
    first `# --- <source>` banner came from a node and must not touch it.
    """
    body = (ROOT / "env" / backend / "requirements.in").read_text()
    _preamble, _, node_section = body.partition("# --- ")
    for line in node_section.splitlines():
        line = line.split("#")[0].strip()
        for pkg in ("torch", "torchvision", "torchaudio"):
            if re.match(rf"^{pkg}\b", line):
                pytest.fail(f"{backend}: node-supplied torch pin leaked through: {line!r}")


@pytest.mark.parametrize("backend", BACKENDS)
def test_backend_pins_torch_to_a_local_version(backend):
    """`torch==2.13.0` resolves to the generic CUDA-linked PyPI wheel even with
    a backend index attached. Only the local tag (+cu130 / +rocm7.1) forces
    resolution onto the backend index. Without it the ROCm image ships a CUDA
    build that imports cleanly and then finds no GPU."""
    expected = LOCK["backends"][backend]["index"].rsplit("/", 1)[-1]
    body = (ROOT / "env" / backend / "requirements.txt").read_text()
    for line in body.splitlines():
        if line.startswith("torch=="):
            assert f"+{expected}" in line, f"{backend}: {line.strip()} lacks +{expected}"
            return
    pytest.fail(f"no torch pin in {backend} lock")


def test_backends_differ_only_in_accelerator_runtime():
    """The claim the whole repo rests on: one definition, two images. If the
    locks diverge outside the accelerator runtime, that claim is false."""
    def packages(backend: str) -> set[str]:
        return {
            line.split("==")[0]
            for line in (ROOT / "env" / backend / "requirements.txt").read_text().splitlines()
            if re.match(r"^[a-z0-9._-]+==", line)
        }

    accelerator = re.compile(r"^(nvidia-|cuda-|triton|pytorch-triton|amd-|rocm)")
    divergent = packages("cuda") ^ packages("rocm")
    unexpected = {p for p in divergent if not accelerator.match(p)}
    assert not unexpected, f"backends diverge outside accelerator runtime: {sorted(unexpected)}"


# --------------------------------------------------------------- build contract

@pytest.mark.parametrize("backend", BACKENDS)
def test_pruned_packages_are_actually_installed(backend):
    """A typo here would prune nothing and be invisible -- the image would just
    stay large. Every named package must exist in the backend's lock."""
    declared = LOCK["backends"][backend].get("prune_packages", [])
    if not declared:
        pytest.skip(f"{backend} prunes no packages")
    installed = {
        line.split("==")[0].lower().replace("_", "-")
        for line in (ROOT / "env" / backend / "requirements.txt").read_text().splitlines()
        if re.match(r"^[a-z0-9._-]+==", line)
    }
    missing = [p for p in declared if p.lower().replace("_", "-") not in installed]
    assert not missing, f"{backend}: prune_packages names nothing installed: {missing}"


def test_prune_runs_in_the_same_layer_as_the_install():
    """Deleting files in a later layer reclaims nothing -- it stacks whiteouts
    on top of the bytes. The prune only saves space if it shares the install's
    RUN instruction."""
    install = next(
        (i for i in containerfile_instructions() if i.startswith("RUN") and "uv pip sync" in i),
        None,
    )
    assert install is not None, "no `uv pip sync` RUN instruction found"
    assert "pin.py prune" in install, "prune must share the install's RUN layer"
    assert "import torch" in install, "the post-prune import guard must run in-build"


def test_uv_version_is_single_sourced():
    """uv builds the venv and compiles the locks, so CI and the image must use
    the same one. The Containerfile's UV_IMAGE default is the second copy of
    that version; this is what stops the two drifting apart silently."""
    declared = LOCK["runtime"]["uv"]
    default = next(
        (i for i in containerfile_instructions() if i.startswith("ARG UV_IMAGE=")), None
    )
    assert default is not None, "Containerfile has no UV_IMAGE arg"
    assert default.strip().endswith(f":{declared}"), (
        f"Containerfile pins uv {default.split(':')[-1]}, manifest declares {declared}"
    )


def test_tree_mutations_share_the_install_layer():
    """Recursive chown/chmod rewrite every file's metadata, so running either
    in a later layer makes overlayfs copy up the whole tree -- shipping a
    second complete copy of the ~3.3GB venv that every host pulls and stores
    for nothing. This regressed once already; the size is invisible in a
    passing build, which is exactly why it needs a test."""
    instructions = containerfile_instructions()
    install = next(
        (i for i in instructions if i.startswith("RUN") and "uv pip sync" in i), None
    )
    assert install is not None, "no `uv pip sync` RUN instruction found"

    for mutation in ("chown -R", "chmod -R", "pin.py fetch"):
        offenders = [
            i for i in instructions
            if i.startswith("RUN") and mutation in i and i is not install
        ]
        assert not offenders, (
            f"`{mutation}` runs in a RUN separate from the install, duplicating "
            f"the tree into another layer: {offenders}"
        )


def test_containerfile_does_not_declare_a_volume_for_durable_state():
    """Regression guard.

    A declared VOLUME makes the engine auto-create an anonymous volume when the
    operator forgets to bind-mount. That satisfies the entrypoint's mountpoint
    check, so an unmounted deployment starts happily and writes the model tree
    into a volume nobody can name. The unmounted case must stay a plain
    directory so it fails loudly.
    """
    body = (ROOT / "containers" / "Containerfile").read_text()
    for line in body.splitlines():
        if line.strip().startswith("VOLUME") and "/var/mnt/diffusion" in line:
            pytest.fail("Containerfile declares VOLUME for durable state; see docstring")


def test_build_reads_only_generated_artifacts():
    """comfyui.toml is intent, not a build input. If the Containerfile starts
    reading it, the image stops being reproducible from the lock alone."""
    offenders = [i for i in containerfile_instructions() if "comfyui.toml" in i]
    assert not offenders, f"Containerfile reads declared intent, not the lock: {offenders}"


def test_dockerignore_excludes_declared_intent():
    """Belt to the above brace: comfyui.toml must not even reach the build context."""
    body = (ROOT / ".dockerignore").read_text()
    assert body.lstrip().startswith("*"), ".dockerignore must deny-by-default"
    allowed = {line[1:] for line in body.splitlines() if line.startswith("!")}
    assert not any("comfyui.toml" in a for a in allowed)


def test_entrypoint_refuses_a_writable_production_environment():
    body = (ROOT / "containers" / "entrypoint.sh").read_text()
    assert "-w " in body and ".venv" in body


def test_image_runs_as_a_non_root_user():
    body = (ROOT / "containers" / "Containerfile").read_text()
    assert re.search(r"^USER comfy$", body, re.M), "runtime stage must drop to comfy"
