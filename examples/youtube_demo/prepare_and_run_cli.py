# examples/youtube_demo/prepare_and_run_cli.py
from __future__ import annotations

from pathlib import Path
import subprocess
import sys

YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=mSwo28t5t3k",
    "https://www.youtube.com/watch?v=Xk3F-1_A9HU",
]

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_DIR = BASE / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


def ensure_yt_dlp() -> None:
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        print("\n[!] Missing dependency: yt-dlp. Install with:  pip install yt-dlp\n", file=sys.stderr)
        raise


def download_youtube(urls: list[str], data_dir: Path) -> None:
    from yt_dlp import YoutubeDL

    ydl_opts = {
        "outtmpl": str(data_dir / "yt_%(id)s.%(ext)s"),
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        for url in urls:
            print(f"[download] {url}")
            try:
                ydl.download([url])
            except Exception as e:
                print(f"[warn] failed to download {url}: {e}", file=sys.stderr)


if __name__ == "__main__":
    ensure_yt_dlp()
    download_youtube(YOUTUBE_URLS, DATA_DIR)

    # Launch TWISTER (opens the GUI, then runs the pipeline)
    cmd = [
        "twister",
        "--videos",
        str(DATA_DIR),
        "--out",
        str(OUT_DIR),
    ]
    print("\n[twister] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)
