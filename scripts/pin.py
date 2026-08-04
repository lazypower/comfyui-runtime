#!/usr/bin/env python3
"""Turn declared intent (manifest/comfyui.toml) into pinned build inputs.

Owns the boundary between "what we want" and "what we build". Three verbs:

  resolve            comfyui.toml        -> manifest/nodes.lock.json   (exact commits)
  collect <backend>  nodes.lock.json     -> env/<backend>/requirements.in
  fetch <dest>       nodes.lock.json     -> git trees at exact commits  (runs inside the build)

`fetch` is the only verb that runs inside the image build, so this file stays
stdlib-only and reads nothing but JSON in that path.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest" / "comfyui.toml"
LOCK = ROOT / "manifest" / "nodes.lock.json"

# The backend owns these. Any node pinning them is overruled.
# NOTE: torchsde is deliberately absent -- it is an arch-neutral SDE solver that
# merely depends on torch, and ComfyUI core requires it. Only packages that ship
# accelerator-specific binaries belong here.
BACKEND_OWNED = {"torch", "torchvision", "torchaudio", "xformers"}


def die(msg: str) -> None:
    print(f"pin: {msg}", file=sys.stderr)
    sys.exit(1)


def load_manifest() -> dict:
    import tomllib  # 3.11+; host-side only

    if not MANIFEST.exists():
        die(f"missing {MANIFEST}")
    return tomllib.loads(MANIFEST.read_text())


def load_lock() -> dict:
    if not LOCK.exists():
        die(f"missing {LOCK} -- run `just lock` first")
    return json.loads(LOCK.read_text())


def slug(repo_url: str) -> str:
    """https://github.com/owner/name(.git) -> owner/name"""
    return repo_url.rstrip("/").removesuffix(".git").split("github.com/", 1)[-1]


# --------------------------------------------------------------------------- resolve


def ls_remote(repo: str, ref: str) -> str:
    """Resolve a tag or branch to its exact commit. Tags are dereferenced."""
    out = subprocess.run(
        ["git", "ls-remote", repo, ref, f"refs/tags/{ref}^{{}}", f"refs/heads/{ref}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    lines = [ln.split("\t") for ln in out.strip().splitlines() if ln.strip()]
    if not lines:
        die(f"{repo}: ref {ref!r} not found")

    # Prefer the peeled tag object (^{}) -- that is the commit a tag points at.
    for sha, name in lines:
        if name.endswith("^{}"):
            return sha
    return lines[0][0]


def cmd_resolve() -> None:
    m = load_manifest()
    core = m["comfyui"]

    print(f"resolving comfyui {core['ref']} ...", file=sys.stderr)
    lock = {
        "_generated_by": "scripts/pin.py resolve -- do not hand-edit",
        "runtime": m["runtime"],
        "backends": m["backends"],
        "comfyui": {
            "repo": core["repo"],
            "ref": core["ref"],
            "commit": ls_remote(core["repo"], core["ref"]),
        },
        "nodes": [],
    }

    for node in m.get("nodes", []):
        print(f"resolving {node['name']} {node['ref']} ...", file=sys.stderr)
        lock["nodes"].append(
            {
                "name": node["name"],
                "repo": node["repo"],
                "ref": node["ref"],
                "commit": ls_remote(node["repo"], node["ref"]),
                **({"role": node["role"]} if "role" in node else {}),
            }
        )

    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"wrote {LOCK.relative_to(ROOT)}", file=sys.stderr)


# --------------------------------------------------------------------------- collect


def fetch_requirements(repo: str, commit: str) -> str | None:
    """Read requirements.txt at an exact commit without cloning."""
    url = f"https://raw.githubusercontent.com/{slug(repo)}/{commit}/requirements.txt"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            return r.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def clean(body: str, source: str) -> list[str]:
    """Strip index directives, includes, and backend-owned packages."""
    kept: list[str] = []
    for raw in body.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):  # --extra-index-url, -r, -e, --find-links
            continue
        name = (
            line.split("[")[0]
            .split("==")[0]
            .split(">=")[0]
            .split("<=")[0]
            .split("~=")[0]
            .split(">")[0]
            .split("<")[0]
            .split("!=")[0]
            .split(";")[0]
            .strip()
            .lower()
            .replace("_", "-")
        )
        if name in BACKEND_OWNED:
            print(f"  overruling {source} pin: {line}", file=sys.stderr)
            continue
        kept.append(line)
    return kept


def latest_backend_build(index: str, package: str, py: str, platform_tag: str = "x86_64") -> str:
    """Newest stable build of `package` published on a PyTorch backend index.

    Needed because a bare `torch` requirement resolves to the generic PyPI wheel
    -- which is CUDA-linked -- even with the ROCm index attached. Pinning the
    exact local version (2.13.0+rocm6.4) is what forces resolution onto the
    backend index, because no other index can satisfy it.
    """
    cp = "cp" + py.replace(".", "")
    url = f"{index}/{package.replace('_', '-')}/"
    with urllib.request.urlopen(url, timeout=60) as r:
        html = r.read().decode("utf-8", "replace")

    pattern = re.compile(
        rf"{re.escape(package)}-([^-]+)-{cp}-{cp}[^\"'>]*?{re.escape(platform_tag)}\.whl"
    )
    versions = {urllib.parse.unquote(m.group(1)) for m in pattern.finditer(html)}
    versions = {v for v in versions if not re.search(r"(dev|rc|a\d|b\d)", v)}
    if not versions:
        die(f"no {package} wheel for {cp}/{platform_tag} on {index}")

    def key(v: str):
        base = v.split("+")[0]
        return tuple(int(p) if p.isdigit() else 0 for p in base.split("."))

    return max(versions, key=key)


def cmd_collect(backend: str) -> None:
    lock = load_lock()
    if backend not in lock["backends"]:
        die(f"unknown backend {backend!r}; have {', '.join(lock['backends'])}")

    sections: list[tuple[str, list[str]]] = []

    core = lock["comfyui"]
    body = fetch_requirements(core["repo"], core["commit"])
    if body is None:
        die("ComfyUI core has no requirements.txt -- upstream layout changed")
    sections.append((f"ComfyUI {core['ref']} ({core['commit'][:12]})", clean(body, "comfyui")))

    for node in lock["nodes"]:
        body = fetch_requirements(node["repo"], node["commit"])
        label = f"{node['name']} ({node['commit'][:12]})"
        if body is None:
            sections.append((f"{label} -- no requirements.txt", []))
            continue
        sections.append((label, clean(body, node["name"])))

    out = ROOT / "env" / backend / "requirements.in"
    out.parent.mkdir(parents=True, exist_ok=True)

    index = lock["backends"][backend]["index"]
    py = lock["runtime"]["python"]

    torch_pins = []
    for pkg in ("torch", "torchvision", "torchaudio"):
        version = latest_backend_build(index, pkg, py)
        print(f"  {backend}: {pkg}=={version}", file=sys.stderr)
        torch_pins.append(f"{pkg}=={version}")

    if not any("+" in p for p in torch_pins):
        die(
            f"{backend}: resolved torch builds carry no local version tag -- they came "
            f"from PyPI, not {index}. The image would ship the wrong accelerator."
        )

    lines = [
        "# GENERATED by `just lock` from manifest/nodes.lock.json -- do not hand-edit.",
        f"# backend: {backend}  index: {index}",
        "",
        "# The backend owns the torch triple, pinned to its LOCAL version so that only",
        "# the backend index can satisfy it. A bare `torch` would resolve to the generic",
        "# PyPI wheel -- CUDA-linked -- and quietly produce a ROCm image with no GPU.",
        *torch_pins,
        "",
    ]
    for label, reqs in sections:
        lines.append(f"# --- {label}")
        lines.extend(reqs if reqs else ["#   (none)"])
        lines.append("")

    out.write_text("\n".join(lines))
    print(f"wrote {out.relative_to(ROOT)}", file=sys.stderr)


# --------------------------------------------------------------------------- fetch


def cmd_fetch(dest: str) -> None:
    """Materialise ComfyUI + pinned nodes at exact commits. Runs inside the build."""
    lock = load_lock()
    root = Path(dest)

    def checkout(repo: str, commit: str, into: Path) -> None:
        into.mkdir(parents=True, exist_ok=True)
        run = lambda *a: subprocess.run(a, cwd=into, check=True)  # noqa: E731
        run("git", "init", "-q")
        run("git", "remote", "add", "origin", repo)
        run("git", "fetch", "-q", "--depth", "1", "origin", commit)
        run("git", "checkout", "-q", "FETCH_HEAD")

    core = lock["comfyui"]
    print(f"fetch comfyui @ {core['commit'][:12]}", file=sys.stderr)
    checkout(core["repo"], core["commit"], root)

    for node in lock["nodes"]:
        print(f"fetch {node['name']} @ {node['commit'][:12]}", file=sys.stderr)
        checkout(node["repo"], node["commit"], root / "custom_nodes" / node["name"])


# --------------------------------------------------------------------------- prune


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n}B"


# Directories whose contents are shipped per GPU architecture. These hold data
# files -- kernel databases and precompiled Tensile objects -- not linked
# libraries, so dropping the ones this host cannot use is safe.
ARCH_DIRS = (
    "torch/share/miopen/db",
    "torch/lib/rocblas/library",
    "torch/lib/hipblaslt/library",
)
GFX = re.compile(r"gfx[0-9a-f]+", re.I)


def cmd_prune(backend: str, venv: str) -> None:
    """Strip what this deployment provably cannot use. Runs inside the build."""
    lock = load_lock()
    config = lock["backends"][backend]
    root = Path(venv)

    site = next(root.glob("lib/python*/site-packages"), None)
    if site is None:
        die(f"no site-packages under {venv}")

    freed = 0

    # 1. whole packages the manifest says are unreachable
    packages = config.get("prune_packages", [])
    if packages:
        print(f"prune: removing {len(packages)} package(s): {', '.join(packages)}", file=sys.stderr)
        subprocess.run(
            ["uv", "pip", "uninstall", "--python", f"{venv}/bin/python", *packages],
            check=True,
        )

    # 2. per-architecture kernel data for cards this image will never see
    archs = {a.lower() for a in config.get("gpu_archs", [])}
    if archs:
        print(f"prune: keeping GPU archs {sorted(archs)}", file=sys.stderr)
        for rel in ARCH_DIRS:
            directory = site / rel
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if not path.is_file():
                    continue
                found = {m.group(0).lower() for m in GFX.finditer(path.name)}
                if found and not (found & archs):
                    freed += path.stat().st_size
                    path.unlink()
        print(f"prune: freed {human(freed)} of foreign-architecture kernels", file=sys.stderr)


# --------------------------------------------------------------------------- main

if __name__ == "__main__":
    match sys.argv[1:]:
        case ["resolve"]:
            cmd_resolve()
        case ["collect", backend]:
            cmd_collect(backend)
        case ["fetch", dest]:
            cmd_fetch(dest)
        case ["prune", backend, venv]:
            cmd_prune(backend, venv)
        case _:
            die("usage: pin.py resolve | collect <backend> | fetch <dest> | prune <backend> <venv>")
