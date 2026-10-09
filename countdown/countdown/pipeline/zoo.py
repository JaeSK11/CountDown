"""The model zoo: where deployed experts come from on a fresh clone (Phase 6, task 6.8).

The artifacts are too big for git (E2 alone is ~330 MB), so they are published as assets of a
GitHub release on this (private) repository and fetched with ``countdown models fetch``.

They are joblib **pickles**, and unpickling a tampered file runs arbitrary code. So the repo
tracks ``models_store.sha256`` -- the SHA-256 of every artifact -- and ``load_store`` refuses
any file whose hash is not in it. Publishing a new zoo means updating that manifest in a commit,
which is what makes the download trustworthy: the hash comes from git, not from the server.

    countdown models fetch            # gh release download <tag> -> models_store/, then verify
    countdown models verify           # check models_store/ against the manifest
    scripts/publish_models.sh v2      # (maintainer) write the manifest + upload a release
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "models_store.sha256"
STORE = ROOT / "models_store"
DEFAULT_TAG = "models-v1"


class UnverifiedModel(RuntimeError):
    """An artifact whose SHA-256 is not the one the repository's manifest records."""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: Path = MANIFEST) -> dict[str, str]:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            digest, name = line.split()
            out[name] = digest
    return out


def write_manifest(store: Path = STORE, path: Path = MANIFEST, tag: str = DEFAULT_TAG) -> Path:
    lines = [f"# SHA-256 of the deployed experts in GitHub release {tag}.",
             "# `countdown models verify` checks models_store/ against this; load_store refuses "
             "anything else."]
    lines += [f"{sha256(f)}  {f.name}" for f in sorted(store.glob("E*.joblib"))]
    path.write_text("\n".join(lines) + "\n")
    return path


def verify(store: Path = STORE, manifest: Path = MANIFEST) -> dict[str, str]:
    """``{file: "ok" | "mismatch" | "missing" | "not in manifest"}``."""
    want = read_manifest(manifest)
    have = {f.name: f for f in sorted(store.glob("E*.joblib"))}
    out = {}
    for name, digest in want.items():
        out[name] = "missing" if name not in have else ("ok" if sha256(have[name]) == digest else "mismatch")
    for name in have:
        out.setdefault(name, "not in manifest")
    return out


def check_file(path: Path, manifest: Path = MANIFEST) -> None:
    """Raise :class:`UnverifiedModel` unless ``path``'s hash is the manifest's.  A store
    without any manifest (a developer's own export) is allowed, with nothing to check."""
    want = read_manifest(manifest)
    if not want:
        return
    if want.get(path.name) != sha256(path):
        raise UnverifiedModel(
            f"{path} does not match {manifest.name} -- refusing to unpickle it. Fetch the "
            f"published zoo (`countdown models fetch`), or for your own export pass "
            f"`--allow-unverified` / regenerate the manifest (scripts/publish_models.sh).")


def _repo() -> str:
    url = subprocess.run(["git", "-C", str(ROOT), "remote", "get-url", "origin"],
                         capture_output=True, text=True, check=True).stdout.strip()
    return url.removesuffix(".git").split("github.com")[-1].lstrip(":/")


def fetch(tag: str = DEFAULT_TAG, store: Path = STORE) -> dict[str, str]:
    """Download the release's artifacts with the GitHub CLI (private repo: uses gh's auth)."""
    store.mkdir(parents=True, exist_ok=True)
    subprocess.run(["gh", "release", "download", tag, "-R", _repo(), "-D", str(store),
                    "-p", "E*.joblib", "--clobber"], check=True)
    return verify(store)
