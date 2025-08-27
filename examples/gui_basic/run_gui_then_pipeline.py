# examples/gui_basic/run_gui_then_pipeline.py
from __future__ import annotations

from pathlib import Path
import os
import sys

from twister.gui.clip_gui import ClipGUI
from twister.io import find_video_files
from twister.twister import twstr

def has_display() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY"))
    return True

def main():
    base = Path(__file__).resolve().parent
    data_dir  = base / "data"          # put your raw videos here
    out_root  = base / "outputs"
    clips_dir = out_root / "clips"
    for d in (data_dir, out_root, clips_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1) Launch GUI to create clips
    app = ClipGUI(video_dir=data_dir, out_dir=clips_dir)
    if has_display():
        app.root.mainloop()
    else:
        print("[warn] No display detected; skipping GUI stage.")

    # Prefer clips if any were saved
    video_root = clips_dir if find_video_files(clips_dir) else data_dir

    # 2) Run pipeline
    tw = twstr(
        video_path=str(video_root),
        output_path=str(out_root),
        plotting_args={"plotting_folder": str(out_root / "plots") + "/", "ext": ".svg"},
    )
    tw.tracking_dir = out_root / "tracking"
    tw.csv_dir = out_root / "csv_predictions"

    tw.run(
        preprocess_videos=False,
        make_video=True,
        save_csv=True,
        recompute_existing=False,
    )

    # 3) Save features
    tw.feature_matrix.to_csv(out_root / "features.csv")
    print(f"[twister] Done. Outputs in {out_root}")

if __name__ == "__main__":
    raise SystemExit(main())
