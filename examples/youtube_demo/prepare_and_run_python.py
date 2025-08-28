# examples/youtube_demo/prepare_and_run_python.py
from __future__ import annotations

from pathlib import Path
import os
import sys

YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=mSwo28t5t3k",
    "https://www.youtube.com/watch?v=Xk3F-1_A9HU",
]

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_ROOT = BASE / "outputs"
CLIPS_DIR = OUT_ROOT / "clips"
for d in (DATA_DIR, OUT_ROOT, CLIPS_DIR):
    d.mkdir(parents=True, exist_ok=True)


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


def has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY"))
    return True


if __name__ == "__main__":
    ensure_yt_dlp()
    download_youtube(YOUTUBE_URLS, DATA_DIR)

    # 1) GUI to create clips (blocking if a display is available)
    from twister.gui.clip_gui import ClipGUI

    app = ClipGUI(video_dir=DATA_DIR, out_dir=CLIPS_DIR)
    if has_display():
        app.root.mainloop()
    else:
        print("[warn] No display detected; skipping GUI stage.")

    # Prefer clips if any were saved; otherwise use raw downloads
    from twister.io import find_video_files

    video_root = CLIPS_DIR if find_video_files(CLIPS_DIR) else DATA_DIR

    # 2) Run the pipeline
    from twister.twister import twstr

    tw = twstr(
        video_path=str(video_root),
        output_path=str(OUT_ROOT),
        plotting_args={"plotting_folder": str(OUT_ROOT / "plots") + "/", "ext": ".svg"},
    )
    # Make prediction layer use explicit output folders
    tw.tracking_dir = OUT_ROOT / "tracking"
    tw.csv_dir = OUT_ROOT / "csv_predictions"

    tw.run(
        make_video=True,
        save_csv=True,
        recompute_existing=False,  # load CSVs if they already exist
    )

    # Save features matrix
    (OUT_ROOT / "features.csv").parent.mkdir(parents=True, exist_ok=True)
    tw.feature_matrix.to_csv(OUT_ROOT / "features.csv")
    print(f"[twister] Done. Outputs in {OUT_ROOT}")
