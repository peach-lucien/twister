from pathlib import Path
from twister.twister import twstr

base = Path(__file__).resolve().parent
data_dir = base / "data"          # put a few mp4s here
out_root = base / "outputs"
out_root.mkdir(parents=True, exist_ok=True)

tw = twstr(
    video_path=str(data_dir),
    output_path=str(out_root),
    plotting_args={"plotting_folder": str(out_root / "plots") + "/", "ext": ".svg"},
)
tw.tracking_dir = out_root / "tracking"
tw.csv_dir = out_root / "csv_predictions"

tw.run(
    preprocess_videos=False,
    make_video=True,
    save_csv=True,
    recompute_existing=False,   # loads cached CSVs if your predictions layer supports it
)

(tw.feature_matrix).to_csv(out_root / "features.csv")
print(f"Done. Outputs in {out_root}")
