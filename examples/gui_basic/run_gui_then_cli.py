# examples/gui_cli_bridge/run_gui_then_cli.py
from __future__ import annotations

from pathlib import Path
import os
import sys
import subprocess

from twister.gui.clip_gui import ClipGUI
from twister.io import find_video_files

def has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY"))
    return True

def main():
    base = Path(__file__).resolve().parent
    data_dir  = base / "data"
    out_root  = base / "outputs"
    clips_dir = out_root / "clips"
    for d in (data_dir, out_root, clips_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) GUI to create clips from raw videos
    app = ClipGUI(video_dir=data_dir, out_dir=clips_dir)
    if has_display():
        app.root.mainloop()
    else:
        print("[warn] No display detected; skipping GUI stage.")

    # 2) Run the CLI WITHOUT GUI on the resulting clips (or raw if none saved)
    video_root = clips_dir if find_video_files(clips_dir) else data_dir
    cmd = [
        "twister",
        "--nogui",
        "--videos", str(video_root),
        "--out",    str(out_root),
        # optional speedups:
        # "--no-make-video",
    ]
    print("[twister] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)

    print(f"[twister] Done. Outputs in {out_root}")

if __name__ == "__main__":
    raise SystemExit(main())
