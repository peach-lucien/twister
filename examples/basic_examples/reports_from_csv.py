# examples/reanalyze_from_csv/reports_from_csv.py
from __future__ import annotations
from pathlib import Path

from twister.twister import twstr
from twister.io import load_from_csv

def main():
    base = Path(__file__).resolve().parent
    out_dir  = base / "outputs"
    csv_dir  = out_dir / "csv_predictions"   # produced by a previous run
    plots_dir = out_dir / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Load patient collection directly from CSV predictions
    pc = load_from_csv(folder=str(csv_dir) + "/")

    # Create a twstr shell just for analysis/plots; no inference
    tw = twstr(video_path=None, output_path=str(out_dir),
               plotting_args={"plotting_folder": str(plots_dir) + "/", "ext": ".svg"})
    tw.patient_collection = pc

    # Compute statistics, aggregate features, and plot reports
    tw.analyse()
    tw.aggregate()
    tw.plot()

    tw.feature_matrix.to_csv(out_dir / "features_from_csv.csv")
    print(f"[twister] re-analysis complete → {out_dir}")

if __name__ == "__main__":
    raise SystemExit(main())
