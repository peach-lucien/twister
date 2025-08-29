# examples/local_minimal/run_pipeline_no_gui.py
from __future__ import annotations
from pathlib import Path

from twister.twister import twstr

def main():
    base = Path(__file__).resolve().parent
    data_dir = base / "data"          # put your videos here
    out_dir  = base / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    tw = twstr(
        video_path=str(data_dir),
        output_path=str(out_dir),
        plotting_args={"plotting_folder": str(out_dir / "plots") + "/", "ext": ".svg"},
    )
    # optional: explicit subfolders for predictions layer
    tw.tracking_dir = out_dir / "tracking"
    tw.csv_dir      = out_dir / "csv_predictions"

    tw.run(
        make_video=False,      # set True to save overlay videos
        save_csv=True,         # write per-model CSVs
        recompute_existing=False,  # re-use CSVs if they exist
    )

    tw.feature_matrix.to_csv(out_dir / "features.csv")
    print(f"[twister] done → {out_dir}")

if __name__ == "__main__":
    raise SystemExit(main())
