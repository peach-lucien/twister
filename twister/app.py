# twister/app.py
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

from twister.twister import twstr
from twister.io import find_video_files
from twister.io import load_from_csv

def _has_display() -> bool:
    # crude check to avoid Tk errors on headless servers
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY"))
    return True  # macOS/Windows usually fine

def _pick_video_root(videos_dir: Path, clips_dir: Path) -> Path:
    """Prefer clips if any exist, else use original videos."""
    clips = find_video_files(str(clips_dir))
    return clips_dir if clips else videos_dir

def _run_pipeline(video_root: Path, out_root: Path, *, make_video: bool,
                  preprocess: bool, recompute_existing: bool, save_csv: bool) -> None:
    out_root = out_root.resolve()
    plots_dir = out_root / "plots"

    # init twstr; set explicit folders so predictions.py uses them
    tw = twstr(
        video_path=str(video_root),
        output_path=str(out_root),
        plotting_args={"plotting_folder": str(plots_dir) + "/", "ext": ".svg"},
    )
    # predictions.py looks for tw.tracking_dir / tw.csv_dir if present
    tw.tracking_dir = out_root / "tracking"
    tw.csv_dir = out_root / "csv_predictions"

    tw.run(
        preprocess=preprocess,
        make_video=make_video,
        save_csv=save_csv,               # passed through to predictions layer
        recompute_existing=recompute_existing,
    )

    # persist features
    (out_root / "features").mkdir(parents=True, exist_ok=True)
    (out_root / "features.csv").write_text(tw.feature_matrix.to_csv())

def cli(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="twister",
        description="Twister pipeline: optionally trim/crop clips via GUI, then run analysis."
    )
    p.add_argument("--videos", type=Path, default=Path("./data"),
                   help="Root folder containing raw videos.")
    p.add_argument("--out", type=Path, default=Path("./outputs"),
                   help="Root output folder (clips, tracking, csv, plots, features).")
    p.add_argument("--clips-subdir", default="clips",
                   help="Subfolder of --out where GUI saves clips (default: clips).")
    p.add_argument("--nogui", action="store_true",
                   help="Skip GUI and run directly on --videos.")
    p.add_argument("--preprocess", action="store_true",
                   help="Run preprocessing step on videos before prediction.")
    p.add_argument("--recompute-existing", action="store_true",
                   help="Force recomputation even if CSVs exist.")
    p.add_argument("--no-make-video", action="store_true",
                   help="Disable saving tracking overlay videos.")
    p.add_argument("--no-save-csv", action="store_true",
                   help="Do not write per-model CSV outputs.")
    args = p.parse_args(argv)

    videos_dir = args.videos.resolve()
    out_root   = args.out.resolve()
    clips_dir  = out_root / args.clips_subdir
    out_root.mkdir(parents=True, exist_ok=True)
    clips_dir.mkdir(parents=True, exist_ok=True)

    video_root = videos_dir

    if not args.nogui:
        if not _has_display():
            print("[twister] No display detected; launching pipeline without GUI. Use --nogui to suppress this message.",
                  file=sys.stderr)
        else:
            # lazy import to avoid Tk dependency when not used
            from twister.gui.clip_gui import ClipGUI
            app = ClipGUI(video_dir=videos_dir, out_dir=clips_dir)
            # If GUI couldn't start (e.g., no videos), it will destroy the root
            try:
                app.root.mainloop()
            except Exception:
                # ensure any dangling OpenCV windows are closed
                try:
                    import cv2
                    cv2.destroyAllWindows()
                except Exception:
                    pass
                raise

            # prefer clips if user saved any
            video_root = _pick_video_root(videos_dir, clips_dir)

    # sanity: ensure there is at least one video to process
    found = find_video_files(str(video_root))
    if not found:
        print(f"[twister] No videos found in {video_root}. Nothing to do.", file=sys.stderr)
        return 2

    _run_pipeline(
        video_root=video_root,
        out_root=out_root,
        make_video=not args.no_make_video,
        preprocess=args.preprocess,
        recompute_existing=args.recompute_existing,
        save_csv=not args.no_save_csv,
    )

    print(f"[twister] Done. Outputs in: {out_root}")
    return 0


def _run_reports_from_csv(csv_dir: Path, out_root: Path, *, ext: str = ".svg") -> None:
    """
    Load prediction CSVs from csv_dir, compute statistics, aggregate features,
    and write reports into out_root.
    """
    out_root = out_root.resolve()
    plots_dir = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load patient collection directly from CSVs
    pc = load_from_csv(folder=str(csv_dir) + "/")

    # 2) Create a twstr shell for analysis/plots; no inference
    t = twstr(
        video_path=None,
        output_path=str(out_root),
        plotting_args={"plotting_folder": str(plots_dir) + "/", "ext": ext},
    )
    t.patient_collection = pc

    # 3) Compute statistics, aggregate features, and plot reports
    t.analyse()
    t.aggregate()
    t.plot()

    # 4) Save features
    (out_root / "features_from_csv.csv").write_text(t.feature_matrix.to_csv())
    print(f"[twister] re-analysis complete → {out_root}")

def cli_from_csv(argv: list[str] | None = None) -> int:
    """
    CLI entrypoint: rebuild reports/features from existing CSV predictions.
    """
    p = argparse.ArgumentParser(
        prog="twister-from-csv",
        description="Generate plots and aggregate features from existing CSV predictions (no inference).",
    )
    p.add_argument("--csv", type=Path, default=None,
                   help="Folder containing CSV predictions. If omitted, defaults to <out>/csv_predictions.")
    p.add_argument("--out", type=Path, default=Path("./outputs"),
                   help="Root output folder (plots, features_from_csv.csv). Default: ./outputs")
    p.add_argument("--ext", default=".svg",
                   help="Plot file extension for embedded figures in reports. Default: .svg")
    args = p.parse_args(argv)

    out_root = args.out.resolve()
    csv_dir = args.csv.resolve() if args.csv else (out_root / "csv_predictions")

    if not csv_dir.exists():
        print(f"[twister] CSV folder not found: {csv_dir}", file=sys.stderr)
        return 2

    _run_reports_from_csv(csv_dir=csv_dir, out_root=out_root, ext=args.ext)
    return 0


if __name__ == "__main__":
    raise SystemExit(cli())
