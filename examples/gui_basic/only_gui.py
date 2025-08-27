# examples/gui_only/clip_only.py
from __future__ import annotations

from pathlib import Path
import os
import sys

from twister.gui.clip_gui import ClipGUI

def has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY"))
    return True

def main():
    base = Path(__file__).resolve().parent
    data_dir  = base / "data"      # raw videos go here
    out_root  = base / "outputs"
    clips_dir = out_root / "clips"
    for d in (data_dir, out_root, clips_dir):
        d.mkdir(parents=True, exist_ok=True)

    app = ClipGUI(video_dir=data_dir, out_dir=clips_dir)
    if has_display():
        app.root.mainloop()
        print(f"[twister] Saved clips in: {clips_dir}")
    else:
        print("[warn] No display detected; GUI not shown. Nothing done.")

if __name__ == "__main__":
    raise SystemExit(main())
