"""The prune decision, against a synthetic site-packages.

This logic decides whether a multi-hundred-megabyte CUDA library gets deleted
from a production image. Its two failure modes are both quiet: prune something
that is linked and `import torch` dies on the GPU host; prune nothing because
every library matched itself and the image merely stays too large. Neither is
visible without a test, and both cost a CI cycle to discover otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from pin import classify_candidates, package_files, soname_references  # noqa: E402


def make_lib(path: Path, *mentions: str) -> None:
    """A stand-in for an ELF: its own soname plus its DT_NEEDED entries, as the
    NUL-separated strings they are in .dynstr."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = b"\x7fELF" + b"\x00".join(m.encode() for m in (path.name, *mentions))
    path.write_bytes(blob)


def make_package(site: Path, name: str, files: list[Path]) -> None:
    dist = site / f"{name.lower().replace('-', '_')}-1.0.dist-info"
    dist.mkdir(parents=True, exist_ok=True)
    lines = [f"{f.relative_to(site)},sha256=x,1" for f in files]
    (dist / "RECORD").write_text("\n".join(lines) + "\n")


@pytest.fixture
def site(tmp_path: Path) -> Path:
    s = tmp_path / "site-packages"

    # A linked dependency: torch references it.
    linked = s / "nvidia" / "linked" / "lib" / "liblinked.so.1"
    make_lib(linked)
    make_package(s, "nvidia-linked", [linked])

    # An unreferenced dependency: nothing but itself mentions it.
    orphan = s / "nvidia" / "orphan" / "lib" / "liborphan.so.2"
    make_lib(orphan)
    make_package(s, "nvidia-orphan", [orphan])

    # Referenced only by another candidate, which is itself unreferenced.
    chained = s / "nvidia" / "chained" / "lib" / "libchained.so.3"
    make_lib(chained)
    make_package(s, "nvidia-chained", [chained])
    make_lib(s / "nvidia" / "orphan" / "lib" / "liborphan_helper.so", "libchained.so.3")

    make_lib(s / "torch" / "lib" / "libtorch_cuda.so", "liblinked.so.1")
    return s


def test_records_who_references_each_soname(site: Path):
    refs = soname_references(site)
    referrers = {p.name for p in refs["liblinked.so.1"]}
    assert "libtorch_cuda.so" in referrers


def test_package_files_reads_the_record(site: Path):
    owned = package_files(site, "nvidia-orphan")
    assert any(p.name == "liborphan.so.2" for p in owned)


def test_keeps_a_package_that_torch_links(site: Path):
    keep, drop = classify_candidates(site, ["nvidia-linked"])
    assert drop == []
    assert keep[0][0] == "nvidia-linked"
    assert "libtorch_cuda.so" in keep[0][2]


def test_drops_a_package_nothing_references(site: Path):
    keep, drop = classify_candidates(site, ["nvidia-orphan"])
    assert drop == ["nvidia-orphan"]
    assert keep == []


def test_self_reference_alone_does_not_save_a_package(site: Path):
    """The regression that would silently disable pruning entirely: a library
    records its own DT_SONAME, so a naive 'mentioned anywhere' test keeps
    everything and the image never shrinks."""
    refs = soname_references(site)
    assert "liborphan.so.2" in refs, "the fixture must mention its own soname"
    _keep, drop = classify_candidates(site, ["nvidia-orphan"])
    assert "nvidia-orphan" in drop


def test_a_reference_from_a_sibling_package_still_counts(site: Path):
    """Conservative on purpose: nvidia-chained is referenced only by a library
    belonging to another candidate. We keep it rather than reason about drop
    ordering -- a missed pruning opportunity is cheap, a broken import is not."""
    keep, drop = classify_candidates(site, ["nvidia-chained"])
    assert drop == []
    assert keep[0][0] == "nvidia-chained"
