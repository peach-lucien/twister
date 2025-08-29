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

**TWISTER** estimates the **T**oronto **W**estern **S**pasmodic **T**orticollis **R**ating Scale from video. It tracks pose/landmarks (MediaPipe), extracts **per-video** feature vectors, and generates **per-video** PDF reports.

- Multiple videos per subject are supported. Each video becomes its own row (e.g. `alice_v0`, `alice_v1`) and its **own PDF**.
- Caching: if CSV outputs already exist, set `--recompute-existing false` (or `recompute_existing=False`) to re-use them.

---

## Installation

### Option A — Conda (recommended for most users)

```bash
# 1) clone
git clone https://github.com/peach-lucien/twister.git
cd twister

# 2) environment with Python 3.11
conda create -n twister python=3.11 -y
conda activate twister

# 3) system ffmpeg (needed for video I/O)
conda install -c conda-forge ffmpeg -y

# 4) install the package (editable)
pip install -e .

# 5) download model assets (MediaPipe tasks + CNN checkpoint)
twister-download-models
```

> **PyTorch (GPU/CPU):** install per the official PyTorch instructions **after** activating the env if you need a specific CUDA build.

### Option B — Plain `venv` + pip

```bash
# 1) clone
git clone https://github.com/peach-lucien/twister.git
cd twister

# 2) create & activate virtualenv
python3 -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate

# 3) ensure ffmpeg is on your system PATH
# macOS (Homebrew):   brew install ffmpeg
# Ubuntu/Debian:      sudo apt-get install ffmpeg
# Windows (choco):    choco install ffmpeg

# 4) install
pip install -e .

# 5) download model assets
twister-download-models
```

**Windows notes**
- Install Git for Windows.
- Install Microsoft **Visual C++ Build Tools** (required by some deps).
- Ensure `ffmpeg.exe` is on your PATH (e.g. via Chocolatey).

**Where models are stored?**  
`twister-download-models` saves to your user cache by default (e.g. `~/.cache/twister/...`). At runtime TWISTER looks in:
1. `$TWISTER_MODELS_DIR` (if set)  
2. the installed package (`twister/models/...`)  
3. the user cache (`~/.cache/twister/...`)

---

## Quick start

### A) CLI with GUI (trim/crop first, then run)

```bash
twister --videos ./path/to/videos --out ./outputs
```

- A GUI opens so you can **select start/end** and optional **ROI** per video.
- Save one or more clips, close the GUI → the pipeline runs on the saved clips (or raw videos if none saved).

**Useful flags**
- `--nogui` — run directly on the folder.
- `--recompute-existing` — recompute predictions even if CSVs exist.
- `--no-make-video` — skip writing overlay/tracking MP4s.
- `--no-save-csv` — skip writing prediction CSVs.

### B) CLI without GUI

```bash
twister --videos ./path/to/videos --out ./outputs --nogui
```

---

## GUI workflow

Open the clip editor directly:

```bash
twister-gui --videos ./path/to/videos --out ./outputs/clips
```

- **Fixed preview** size so controls stay visible on all screens.
- **Timeline** with draggable start/end, red playhead, Play/Pause, step ±1, speed control, jump to start/end.
- **ROI** selection via OpenCV; **Clear ROI** to reset.
- **Rotate** ±90° if your videos are side-on.
- Keyboard: `space` (play/pause), `s` (set start), `e` (set end), `←/→` (step).

Clips are saved like: `name__s<start>_e<end>.mp4` under your chosen `--out` folder.

---

## Run from Python

Minimal example (equivalent to the CLI):

```python
from pathlib import Path
from twister.twister import twstr

data_dir = Path("./data")
out_dir  = Path("./outputs")
out_dir.mkdir(parents=True, exist_ok=True)

tw = twstr(
    video_path=str(data_dir),
    output_path=str(out_dir),
    plotting_args={"plotting_folder": str(out_dir / "plots") + "/", "ext": ".svg"},
)
tw.tracking_dir = out_dir / "tracking"          # optional: explicit output subdirs
tw.csv_dir      = out_dir / "csv_predictions"

tw.run(
    preprocess_videos=False,
    make_video=True,
    save_csv=True,
    recompute_existing=False,   # re-use cached CSVs if present
)

tw.feature_matrix.to_csv(out_dir / "features.csv")
print("Done →", out_dir)
```

> Want the GUI from Python? Import and launch `ClipGUI` first, then point `video_path` to the clips folder.

---

## Project layout & outputs

```
twister/
  app.py                   # CLI entrypoint (twister)
  twister.py               # pipeline orchestrator (twstr)
  gui/clip_gui.py          # GUI
  models/
    mediapipe_models/      # *.task (downloaded)
    movement_models/       # model_multilabel.pth (downloaded)
  statistics/, plotting/, videos/, ...
examples/
  youtube_demo/, gui_basic/, ...
```

Pipeline results under `--out`:

```
outputs/
  clips/                   # GUI-saved subclips (if any)
  tracking/                # overlay videos (if enabled)
  csv_predictions/         # per-model CSVs
  artifacts/               # checkpoints (pickles)
  plots/<patient_id>/      # per-video PDFs
  features.csv             # one row per video
```

---

## FAQ / Troubleshooting

**`twister: command not found`**  
Activate your environment or re-install in the current shell:
```bash
conda activate twister           # or: source .venv/bin/activate
pip install -e .
```

**`ModuleNotFoundError: twister.gui`**  
Make sure `twister/gui/clip_gui.py` exists and both `twister/` and `twister/gui/` have `__init__.py`. Then:
```bash
pip uninstall -y twister
pip install -e .
```

**MediaPipe model files not found**  
Run:
```bash
twister-download-models
```
(or set `TWISTER_MODELS_DIR=/path/to/models_base`)

**FFmpeg / video I/O issues**  
Install a system `ffmpeg` and ensure it’s on your PATH. If using Conda and you hit conflicts:
```bash
conda remove --force ffmpeg
# then install a system ffmpeg (brew/apt/choco) or re-install from conda-forge cleanly
```

---

## License

MIT — see [LICENSE](./LICENSE).

