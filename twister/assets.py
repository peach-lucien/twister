# twister/assets.py
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen, Request

import importlib.resources as ir
from platformdirs import user_cache_dir
from tqdm.auto import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Asset registry
# name -> (URL, relative_subpath_under_models_base, sha256 or None)
# The base directory is: <base>/<subpath>, where <base> is chosen by --dir
#  • --dir package → site-packages/twister/models
#  • --dir cache   → ~/.cache/twister/mediapipe_models (and siblings)
#  • --dir custom  → user-supplied base dir
# ─────────────────────────────────────────────────────────────────────────────

ASSETS: dict[str, tuple[str, str, str | None]] = {
    # MediaPipe task files
    "hand_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        "mediapipe_models/hand_landmarker.task",
        None,
    ),
    "face_landmarker": (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
        "mediapipe_models/face_landmarker.task",
        None,
    ),
    "pose_landmarker_heavy": (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
        "mediapipe_models/pose_landmarker_heavy.task",
        None,
    ),

    # CNN model (Dataverse)
    # Stores as: twister/models/movement_models/model_multilabel.pth
    "cnn_multilabel": (
        "https://dataverse.harvard.edu/api/access/datafile/8542960",
        "movement_models/model_multilabel.pth",
        None,  # add sha256 string here if you want integrity checking
    ),
}

# Group aliases for convenience
GROUPS: dict[str, list[str]] = {
    "mediapipe": ["hand_landmarker", "face_landmarker", "pose_landmarker_heavy"],
    "movement": ["cnn_multilabel"],
    "all": list(ASSETS.keys()),
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = Request(url, headers={"User-Agent": "twister/1.0"})
    with urlopen(req) as r:
        total = int(r.headers.get("Content-Length", 0))
        with tmp.open("wb") as f, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=dest.name
        ) as p:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                if total:
                    p.update(len(chunk))
    tmp.replace(dest)


def _package_models_base() -> Path:
    # Base folder: site-packages/twister/models
    return Path(ir.files("twister")) / "models"


def _cache_models_base() -> Path:
    # Base folder: ~/.cache/twister
    # We'll store under ~/.cache/twister/<subdir> (e.g. mediapipe_models/, movement_models/)
    return Path(user_cache_dir("twister", "twister"))


def _is_writable_dir(p: Path) -> bool:
    try:
        p.mkdir(parents=True, exist_ok=True)
        t = p / ".twister_write_test"
        t.write_text("ok")
        t.unlink()
        return True
    except Exception:
        return False


def _resolve_base_dir(which: str, custom: Path | None) -> Path:
    if which == "package":
        base = _package_models_base()
        if _is_writable_dir(base):
            return base
        print(f"[twister] package dir not writable: {base} — falling back to cache")
        return _cache_models_base()
    if which == "cache":
        return _cache_models_base()
    if which == "custom":
        assert custom is not None, "--path is required with --dir custom"
        custom.mkdir(parents=True, exist_ok=True)
        return custom
    raise ValueError(which)


def _expand_names(names: list[str]) -> list[str]:
    if not names:
        return GROUPS["all"]
    out: list[str] = []
    for n in names:
        if n in GROUPS:
            out.extend(GROUPS[n])
        elif n in ASSETS:
            out.append(n)
        else:
            raise SystemExit(f"Unknown asset/model '{n}'. Choices: {list(ASSETS)} or groups {list(GROUPS)}")
    # de-dup preserving order
    seen = set()
    uniq = []
    for n in out:
        if n not in seen:
            uniq.append(n); seen.add(n)
    return uniq


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def download_models(names: list[str], base_dir: Path, *, force: bool = False) -> list[Path]:
    targets: list[Path] = []
    for name in _expand_names(names):
        url, rel_subpath, expected = ASSETS[name]
        dest = base_dir / rel_subpath
        if dest.exists() and not force:
            if expected:
                got = _sha256(dest)
                if got != expected:
                    print(f"[twister] {rel_subpath} hash mismatch; re-downloading…")
                else:
                    print(f"[twister] already present: {dest}")
                    targets.append(dest)
                    continue
            else:
                print(f"[twister] already present: {dest}")
                targets.append(dest)
                continue

        print(f"[twister] downloading {name} → {dest}")
        _download(url, dest)
        if expected:
            got = _sha256(dest)
            if got != expected:
                dest.unlink(missing_ok=True)
                raise SystemExit(f"Hash verification failed for {rel_subpath}")
        targets.append(dest)
    return targets


def cli_download_models(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Download model assets for TWISTER (MediaPipe tasks & CNN checkpoint)")
    ap.add_argument("--dir", choices=["package", "cache", "custom"], default="cache",
                    help="Where to place files (default: cache). 'package' tries site-packages, falls back to cache.")
    ap.add_argument("--path", type=Path, default=None,
                    help="Used with --dir custom to specify a base directory.")
    ap.add_argument("--force", action="store_true", help="Re-download even if present.")
    ap.add_argument(
        "models",
        nargs="*",
        default=["all"],
        help=f"Subset to download. Options: {list(ASSETS)} or groups {list(GROUPS)}; default: all",
    )
    args = ap.parse_args(argv)

    base = _resolve_base_dir(args.dir, args.path)
    paths = download_models(args.models, base, force=args.force)

    print("\n[twister] downloaded:")
    for p in paths:
        print("  -", p)

    print(
        "\nRuntime lookup order:\n"
        "  1) TWISTER_MODELS_DIR (base) → e.g. $TWISTER_MODELS_DIR/mediapipe_models/*.task\n"
        "  2) package dir (twister/models/...)\n"
        "  3) user cache (~/.cache/twister/...)\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(cli_download_models())
