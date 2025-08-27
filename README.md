<div align="center">

<p align="center">
  <img src="./artwork/twister_header.png" alt="twister header">
</p>

[🛠️ Installation](#installation) •
[▶️ Quick start](#quick-start) •
[🖼️ GUI workflow](#gui-workflow) •
[🧪 Run from Python](#run-from-python) •
[📦 Project layout & outputs](#project-layout--outputs) •
[❓FAQ / Troubleshooting](#faq--troubleshooting)

![License: MIT](https://img.shields.io/badge/license-MIT-blue)
[![Twitter Follow](https://shields.io/twitter/follow/RobertPeach15.svg)](https://twitter.com/RobertPeach15)

</div>

# TWISTER

**TWISTER** estimates the **T**oronto **W**estern **S**pasmodic **T**orticollis **R**ating Scale from video. It tracks pose/landmarks (MediaPipe), extracts per-video feature vectors, and generates per-video PDF reports.

- Multiple videos per subject are supported. Each video becomes its own row (e.g. `alice_v0`, `alice_v1`), and a **separate PDF** is produced per video.
- Computation is cache-aware: when CSV outputs already exist, they are loaded back into memory (no recomputation) if requested.

---

## Installation


```bash
git clone https://github.com/peach-lucien/twister.git
conda create -n twister python=3.11 -y
conda activate twister
conda install -c conda-forge ffmpeg -y
cd twister
pip install -e .
```

### Dependencies

`pip install -e .` will pull Python deps (numpy, pandas, torch, mediapipe==0.10.11, etc.).

System notes:

- **FFmpeg:** OpenCV video IO on Linux/macOS typically needs system ffmpeg. Prefer system ffmpeg over Conda’s:
  ```bash
  conda remove --force ffmpeg   # if Conda’s ffmpeg causes issues
  ```
- **MediaPipe task files:** the following files must be present under the package path
  `twister/models/mediapipe_models/`:
  - `hand_landmarker.task` (https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
  - `face_landmarker.task` (https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)
  - `pose_landmarker_heavy.task` (https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task)

  Download the official `.task` files and place them inside the mediapipe_models folder (same filenames). TWISTER will read them from that directory at runtime.

---

## Quick start

### A) CLI with GUI (trim/crop first, then run)

```bash
twister --videos ./examples/example_2/data --out ./examples/example_2/outputs
```

- A GUI opens so you can **select start/end** and **optional ROI** per video.
- Save one or more clips, close the GUI. The pipeline then runs on the saved clips
  (or on the original videos if no clips were saved).

Common flags:

- `--nogui` — run directly on `--videos` without the GUI.
- `--preprocess` — run the preprocessor to produce smaller intermediate videos.
- `--recompute-existing` — ignore cached CSVs and recompute predictions.
- `--no-make-video` — don’t save tracking overlay videos.
- `--no-save-csv` — don’t write per-model CSV outputs.

### B) CLI without GUI

```bash
twister --videos ./data --out ./outputs --nogui
```

---

## GUI workflow

Launch directly if you just want to create clips:

```bash
twister-gui --videos ./data --out ./outputs/clips
```

Features:

- **Video list** with FPS / frames / duration / **clip count** (processed videos are tinted).
- **Single timeline** with shaded **start–end** region, **draggable handles**, and **red playhead**.
- **Set Start (s)**, **Set End (e)** buttons (or press keyboard shortcuts `s`, `e`).
- **ROI selection** via OpenCV picker; **Clear ROI** to reset.
- Keyboard: `space` play/pause, `←/→` step frames.

Clips are saved as: `name__s<start>_e<end>.mp4` in the chosen `--out` folder.

---

## Run from Python

Example script (equivalent to the CLI), including GUI stage:

```python
from pathlib import Path
from twister.twister import twstr
from twister.io import find_video_files

# Paths
base = Path(__file__).resolve().parent
data_dir  = base / "data"
out_root  = base / "outputs"
clips_dir = out_root / "clips"
clips_dir.mkdir(parents=True, exist_ok=True)

# 1) GUI: create clips (blocking)
from twister.gui.clip_gui import ClipGUI
app = ClipGUI(video_dir=data_dir, out_dir=clips_dir)
app.root.mainloop()

# Prefer clips if any saved
video_root = clips_dir if find_video_files(clips_dir) else data_dir

# 2) Run pipeline
tw = twstr(
    video_path=str(video_root),
    output_path=str(out_root),
    plotting_args={"plotting_folder": str(out_root / "plots") + "/", "ext": ".svg"},
)
# predictions.py will use these if present
tw.tracking_dir = out_root / "tracking"
tw.csv_dir = out_root / "csv_predictions"

tw.run(
    preprocess_videos=False,
    make_video=True,
    save_csv=True,
    recompute_existing=False,   # load CSVs if present
)

# 3) Save features
(tw.output_path if hasattr(tw,"output_path") else out_root).mkdir(parents=True, exist_ok=True)
(tw.feature_matrix).to_csv(out_root / "features.csv")
```

---

## Project layout & outputs

Typical repo layout (subset):

```
twister/
  app.py                     # CLI entrypoint
  twister.py                 # twstr class (pipeline orchestrator)
  io.py
  gui/                       # GUI
  models/                    # *.task files
  statistics/                # feature extraction
  plotting/                  # report plots
  videos/
examples/
  some_example_folder/
    data/                    # raw videos
    outputs/                 # created by you
```

Pipeline outputs under `--out`:

```
outputs/
  clips/                     # GUI-saved subclips (if any)
  tracking/                  # overlay videos (if enabled)
  csv_predictions/           # per-model CSVs
  artifacts/                 # periodic checkpoints (pickles)
  plots/<patient_id>/        # per-video PDFs go here
  features.csv               # aggregated feature matrix (one row per video)
```

**Multiple videos per patient:** each video is treated independently and indexed as
`<patient_id>_v<k>`, e.g. `alice_v0`, `alice_v1`. Plots are emitted as
`analysis_report_<patient_id>_v<k>.pdf`.

**Caching:** with `recompute_existing=False`, existing CSVs are **loaded** back into the
object (no inference), so downstream statistics/plots run without recomputation.

---

## YouTube demo: download sample videos and run TWISTER GUI

This example uses **yt-dlp** to fetch two public YouTube videos, stores them under `examples/youtube_demo/data/`, and then runs the **TWISTER** pipeline with the GUI, both via the **CLI** and directly from **Python**.

> Requirements: `yt-dlp` (and typically `ffmpeg`). Install with:
>
> ```bash
> pip install yt-dlp
> # optional but recommended for robust merging/conversion
> brew install ffmpeg          # macOS (Homebrew)
> sudo apt-get install ffmpeg  # Ubuntu/Debian
> ```

---

### Script 1 — Run via CLI (launches the GUI automatically)

**File:** `examples/youtube_demo/prepare_and_run_cli.py`

```python
from __future__ import annotations
from pathlib import Path
import subprocess
import sys

# ---- config ----
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=mSwo28t5t3k",
    "https://www.youtube.com/watch?v=Xk3F-1_A9HU",
]

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
OUT_DIR  = BASE / "outputs"
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- helpers ----

def ensure_yt_dlp():
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        print("\n[!] Missing dependency: yt-dlp. Install with:  pip install yt-dlp\n", file=sys.stderr)
        raise


def download_youtube(urls: list[str], data_dir: Path):
    from yt_dlp import YoutubeDL
    # Save as yt_<id>.mp4 under data_dir
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
            try:
                print(f"[download] {url}")
                ydl.download([url])
            except Exception as e:
                print(f"[warn] failed to download {url}: {e}", file=sys.stderr)


if __name__ == "__main__":
    ensure_yt_dlp()
    download_youtube(YOUTUBE_URLS, DATA_DIR)

    # Run the TWISTER CLI (opens the GUI first, then runs the pipeline)
    cmd = [
        "twister",
        "--videos", str(DATA_DIR),
        "--out",    str(OUT_DIR),
    ]
    print("\n[twister] launching:", " ".join(cmd))
    subprocess.run(cmd, check=True)
```

**Usage:**

```bash
python examples/youtube_demo/prepare_and_run_cli.py
```

This will download the two videos into `examples/youtube_demo/data/`, open the TWISTER GUI so you can trim/crop clips, close the GUI, and then run the pipeline. Results go to `examples/youtube_demo/outputs/`.

---

### Script 2 — Run from Python (explicitly call GUI + pipeline)

**File:** `examples/youtube_demo/prepare_and_run_python.py`

```python
from __future__ import annotations
from pathlib import Path
import os
import sys

# ---- config ----
YOUTUBE_URLS = [
    "https://www.youtube.com/watch?v=mSwo28t5t3k",
    "https://www.youtube.com/watch?v=Xk3F-1_A9HU",
]

BASE = Path(__file__).resolve().parent
DATA_DIR  = BASE / "data"
OUT_ROOT  = BASE / "outputs"
CLIPS_DIR = OUT_ROOT / "clips"
for d in (DATA_DIR, OUT_ROOT, CLIPS_DIR):
    d.mkdir(parents=True, exist_ok=True)


def ensure_yt_dlp():
    try:
        import yt_dlp  # noqa: F401
    except Exception:
        print("\n[!] Missing dependency: yt-dlp. Install with:  pip install yt-dlp\n", file=sys.stderr)
        raise


def download_youtube(urls, data_dir: Path):
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
            try:
                print(f"[download] {url}")
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

    # 1) Launch GUI to create clips
    from twister.gui.clip_gui import ClipGUI
    app = ClipGUI(video_dir=DATA_DIR, out_dir=CLIPS_DIR)
    if has_display():
        app.root.mainloop()
    else:
        print("[warn] No display detected; skipping GUI stage.")

    # Prefer clips if any were saved
    from twister.io import find_video_files
    video_root = CLIPS_DIR if find_video_files(CLIPS_DIR) else DATA_DIR

    # 2) Run the pipeline
    from twister.twister import twstr

    tw = twstr(
        video_path=str(video_root),
        output_path=str(OUT_ROOT),
        plotting_args={"plotting_folder": str(OUT_ROOT / "plots") + "/", "ext": ".svg"},
    )
    # directories used by predictions layer if present
    tw.tracking_dir = OUT_ROOT / "tracking"
    tw.csv_dir = OUT_ROOT / "csv_predictions"

    tw.run(
        preprocess_videos=False,
        make_video=True,
        save_csv=True,
        recompute_existing=False,
    )

    # Save features matrix
    (OUT_ROOT / "features.csv").write_text(tw.feature_matrix.to_csv())
    print(f"[twister] Done. Outputs in {OUT_ROOT}")
```

**Usage:**

```bash
python examples/youtube_demo/prepare_and_run_python.py
```

This script performs the same steps as the CLI flow but runs everything from Python: download → GUI → pipeline → features.

---

### Notes

- The example assumes H.264/AAC MP4 outputs. If yt-dlp chooses a split video/audio stream, it will attempt to merge and may require a system `ffmpeg`.
- If you prefer `pytube`, you can swap the downloader, but `yt-dlp` is typically more robust for format selection/merging.
- The TWISTER GUI saves clips as `name__s<start>_e<end>.mp4` under `outputs/clips/`. The pipeline then treats each clip as a separate video.




---

## Reference

- **Publication:** Peach, R., Friedrich, M., Fronemann, L. …, Ip CW. Head movement dynamics in dystonia: a multi-centre retrospective study using visual perceptive deep learning. npj Digit. Med. 7, 160 (2024). https://doi.org/10.1038/s41746-024-01140-6

---

## FAQ / Troubleshooting

**`ModuleNotFoundError: No module named 'twister.gui'`**  
Ensure the file is at `twister/gui/clip_gui.py` and both `twister/` and `twister/gui/`
contain `__init__.py`. Then reinstall:
```bash
pip uninstall -y twister
pip install -e .
```

**GUI won’t open on a server**  
Run without GUI:
```bash
twister --nogui --videos ./data --out ./outputs
```

**MediaPipe model files not found**  
Place `hand_landmarker.task`, `face_landmarker.task`,
`pose_landmarker_heavy.task` under `twister/models/mediapipe_models/`.


---

## License

MIT — see [LICENSE](./LICENSE).

