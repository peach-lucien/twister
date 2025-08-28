# twister/gui/clip_gui.py
from __future__ import annotations
import argparse
import os
import re
import time
from pathlib import Path
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

# ───────── config ─────────
VIDEO_EXTS  = (".mp4", ".MP4", ".mov", ".MOV", ".avi", ".mkv")
CLIP_RE     = re.compile(r"^(?P<stem>.+)__s(?P<s>\d+)_e(?P<e>\d+)\.mp4$")
TIMELINE_H  = 28
HANDLE_W    = 8
PREVIEW_W   = 800    # fixed preview size (keeps UI stable)
PREVIEW_H   = 450

PLAY_SPEEDS = ["0.25x", "0.5x", "1x", "1.5x", "2x"]
SPEED_MAP   = {"0.25x": 0.25, "0.5x": 0.5, "1x": 1.0, "1.5x": 1.5, "2x": 2.0}

# ───────── fs helpers ─────────

def list_videos(root: Path) -> list[Path]:
    vids: list[Path] = []
    for r, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(VIDEO_EXTS) and "preprocessed" not in f:
                vids.append(Path(r) / f)
    return sorted(vids)

def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def probe_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n, w, h

def list_clips_for(stem: str, out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    hits = []
    for p in out_dir.glob(f"{stem}__s*_e*.mp4"):
        if CLIP_RE.match(p.name):
            hits.append(p)
    return sorted(hits)

def write_clip(src: Path, dst_dir: Path, start_f: int, end_f: int,
               roi: tuple[int,int,int,int] | None,
               fourcc: str = "mp4v") -> Path:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open {src}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start  = max(0, min(start_f, n_fr - 1))
    end    = max(start + 1, min(end_f, n_fr))

    if roi:
        x, y, w, h = roi
        out_w, out_h = int(w), int(h)
    else:
        x = y = 0
        out_w, out_h = width, height

    safe_mkdir(dst_dir)
    out = dst_dir / f"{src.stem}__s{start}_e{end}.mp4"
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*fourcc), fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Cannot write to {out}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        if roi:
            frame = frame[y:y+out_h, x:x+out_w]
        writer.write(frame)

    writer.release()
    cap.release()
    return out

# ───────── GUI ─────────

class ClipGUI:
    def __init__(self, video_dir: Path, out_dir: Path):
        self.root = tk.Tk()
        self.root.title("Twister — Clip Selector")
        self.root.minsize(1100, 720)  # enforce enough space for controls

        try:
            style = ttk.Style(self.root)
            if "clam" in style.theme_names():
                style.theme_use("clam")
            style.configure("TButton", padding=6)
            style.configure("TLabel", padding=2)
            style.configure("Treeview", rowheight=24)
        except Exception:
            pass

        self.video_dir = Path(video_dir)
        self.out_dir   = Path(out_dir)

        self.videos = list_videos(self.video_dir)
        if not self.videos:
            messagebox.showerror("No videos", f"No videos found under {self.video_dir}")
            self.root.destroy()
            return

        # state
        self.idx = 0
        self.cap = None
        self.fps = 30.0
        self.n_frames = 0
        self.width = self.height = 0
        self.roi: tuple[int,int,int,int] | None = None

        # selection/playback
        self.start_var = tk.IntVar(value=0)
        self.end_var   = tk.IntVar(value=0)
        self._drag_target: str | None = None
        self._playing = False
        self._play_after_id: str | None = None
        self.play_speed = 1.0

        # frame image ref (Tk must keep a reference)
        self._preview_photo: ImageTk.PhotoImage | None = None

        self._build()
        self._populate_tree()
        self._select_row_by_index(0)

    # ── UI build

    def _build(self):
        # top bar
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Open Folder…", command=self._choose_folder).pack(side="left")
        ttk.Button(bar, text="Choose Output…", command=self._choose_out).pack(side="left", padx=(6,0))
        self.path_label = ttk.Label(bar, text=f"Videos: {self.video_dir}   |   Output: {self.out_dir}")
        self.path_label.pack(side="left", padx=12)

        # main split
        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # left: list
        left = ttk.Frame(paned)
        self.tree = ttk.Treeview(left, columns=("name","fps","frames","dur","clips"),
                                 show="headings", selectmode="browse")
        for col, label, width, anchor in (
            ("name","Video", 280, "w"),
            ("fps","FPS", 60, "e"),
            ("frames","Frames", 80, "e"),
            ("dur","Duration (s)", 110, "e"),
            ("clips","Clips", 60, "e"),
        ):
            self.tree.heading(col, text=label)
            self.tree.column(col, width=width, anchor=anchor)
        yscroll = ttk.Scrollbar(left, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=yscroll.set)
        self.tree.pack(side="left", fill="both", expand=True)
        yscroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)
        self.tree.bind("<Double-1>", self._on_tree_double)
        paned.add(left, weight=1)

        # right: viewer + controls
        right = ttk.Frame(paned)
        paned.add(right, weight=3)

        self.title_label = ttk.Label(right, text="—", font=("TkDefaultFont", 12, "bold"))
        self.title_label.pack(anchor="w", pady=(4,2))

        # fixed-size preview canvas
        self.preview = tk.Canvas(right, width=PREVIEW_W, height=PREVIEW_H,
                                 bg="#111", highlightthickness=1, highlightbackground="#333")
        self.preview.pack(pady=4)

        # info line
        self.info = ttk.Label(right, text="—")
        self.info.pack()

        # scrub (playhead)
        self.scrub = ttk.Scale(right, from_=0, to=100, orient="horizontal", command=self._on_scrub)
        self.scrub.pack(fill="x", padx=4, pady=(6,2))

        # timeline (ROI + handles + playhead)
        self.timeline = tk.Canvas(right, height=TIMELINE_H, background="#f5f7fb",
                                  highlightthickness=1, highlightbackground="#ccd")
        self.timeline.pack(fill="x", padx=4, pady=(0,8))
        self.timeline.bind("<Configure>", lambda _e: self._draw_timeline())
        self.timeline.bind("<Button-1>", self._timeline_click)
        self.timeline.bind("<B1-Motion>", self._timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self._timeline_release)

        # playback controls
        controls = ttk.Frame(right)
        controls.pack(fill="x", pady=(4,2))

        self.play_btn = ttk.Button(controls, text="▶ Play", command=self._toggle_play)
        self.play_btn.pack(side="left")

        ttk.Button(controls, text="⟲ Start", command=self._jump_start).pack(side="left", padx=(8,0))
        ttk.Button(controls, text="⟶ End",   command=self._jump_end).pack(side="left", padx=(4,0))
        ttk.Button(controls, text="◀ Step",  command=lambda: self._nudge(-1)).pack(side="left", padx=(12,0))
        ttk.Button(controls, text="Step ▶",  command=lambda: self._nudge(+1)).pack(side="left", padx=(4,0))

        ttk.Label(controls, text="Speed").pack(side="left", padx=(16,4))
        self.speed_var = tk.StringVar(value="1x")
        self.speed_combo = ttk.Combobox(controls, state="readonly", width=6,
                                        textvariable=self.speed_var, values=PLAY_SPEEDS)
        self.speed_combo.pack(side="left")
        self.speed_combo.bind("<<ComboboxSelected>>", self._on_speed_change)

        ttk.Button(controls, text="Select ROI", command=self._select_roi).pack(side="left", padx=(16,0))
        ttk.Button(controls, text="Clear ROI",  command=self._clear_roi).pack(side="left", padx=(4,0))

        actions = ttk.Frame(right)
        actions.pack(fill="x", pady=(6,8))
        ttk.Button(actions, text="Set Start (s)", command=self._set_start_current).pack(side="right", padx=(6,0))
        ttk.Button(actions, text="Set End (e)",   command=self._set_end_current).pack(side="right", padx=(6,0))
        ttk.Button(actions, text="Save Clip",     command=self._save_clip).pack(side="right", padx=(12,0))

        # status bar
        self.status = ttk.Label(self.root, relief="sunken", anchor="w")
        self.status.pack(fill="x", padx=8, pady=(0,8))

        # keyboard shortcuts
        self.root.bind("<space>", lambda _e: self._toggle_play())
        self.root.bind("s",      lambda _e: self._set_start_current())
        self.root.bind("e",      lambda _e: self._set_end_current())
        self.root.bind("<Left>", lambda _e: self._nudge(-1))
        self.root.bind("<Right>",lambda _e: self._nudge(+1))

    # ── list / load

    def _populate_tree(self):
        self.tree.delete(*self.tree.get_children())
        for i, p in enumerate(self.videos):
            try:
                fps, n, _w, _h = probe_video(p)
                dur = n / fps if fps > 0 else 0.0
            except Exception:
                fps, n, dur = 0.0, 0, 0.0
            clips = list_clips_for(p.stem, self.out_dir)
            tag = "done" if clips else "todo"
            self.tree.insert("", "end", iid=str(i),
                             values=(p.name, f"{fps:.1f}", f"{n}", f"{dur:.2f}", str(len(clips))),
                             tags=(tag,))
        self.tree.tag_configure("done", background="#e9f7ef")
        self.tree.tag_configure("todo", background="#fff")

    def _on_tree_select(self, _evt):
        sel = self.tree.selection()
        if not sel:
            return
        self._load_video(int(sel[0]))

    def _on_tree_double(self, _evt):
        self._on_tree_select(_evt)

    def _select_row_by_index(self, i: int):
        i = max(0, min(i, len(self.videos)-1))
        self.tree.selection_set(str(i))
        self.tree.see(str(i))

    def _open_cap(self, path: Path) -> bool:
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Cannot open {path}")
            return False
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.n_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width  = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def _load_video(self, i: int):
        if not self.videos:
            return
        self.idx = max(0, min(i, len(self.videos) - 1))
        path = self.videos[self.idx]
        if not self._open_cap(path):
            return
        self.title_label.config(text=str(path.name))
        self.scrub.configure(to=self.n_frames - 1)
        self.start_var.set(0)
        self.end_var.set(self.n_frames - 1)
        self.scrub.set(0)
        self.roi = None
        self._show_frame(0)
        self._update_status()
        self._draw_timeline()

    # ── preview / playback

    def _read_frame(self, idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        return ok, frame

    def _render_on_canvas(self, im: Image.Image):
        # letterbox into fixed PREVIEW_W x PREVIEW_H
        bw, bh = PREVIEW_W, PREVIEW_H
        bg = Image.new("RGB", (bw, bh), (17, 17, 17))
        # compute aspect-fit size
        scale = min(bw / im.width, bh / im.height)
        nw, nh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        im2 = im.resize((nw, nh))
        x0 = (bw - nw) // 2
        y0 = (bh - nh) // 2
        bg.paste(im2, (x0, y0))
        self._preview_photo = ImageTk.PhotoImage(bg)
        self.preview.create_image(0, 0, image=self._preview_photo, anchor="nw")

    def _show_frame(self, idx: int):
        ok, frame = self._read_frame(idx)
        if not ok:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im  = Image.fromarray(rgb)
        self._render_on_canvas(im)
        t = idx / self.fps
        self.info.config(text=f"frame {idx}/{self.n_frames-1}  ({t:.2f}s @ {self.fps:.2f} fps)")
        self._draw_timeline()  # update playhead line

    def _on_scrub(self, val):
        self._show_frame(int(float(val)))

    def _nudge(self, delta: int):
        new = max(0, min(self.n_frames - 1, int(self.scrub.get()) + delta))
        self.scrub.set(new)
        self._show_frame(new)

    def _on_speed_change(self, _evt=None):
        self.play_speed = SPEED_MAP.get(self.speed_var.get(), 1.0)

    def _toggle_play(self):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        self._playing = True
        self.play_btn.config(text="⏸ Pause")
        self._on_speed_change()
        self._play_tick()

    def _stop_playback(self):
        self._playing = False
        self.play_btn.config(text="▶ Play")
        if self._play_after_id is not None:
            try:
                self.root.after_cancel(self._play_after_id)
            except Exception:
                pass
            self._play_after_id = None

    def _play_tick(self):
        if not self._playing:
            return
        i = int(self.scrub.get())
        if i >= self.n_frames - 1:
            self._stop_playback()
            return
        self._nudge(+1)
        delay_ms = max(1, int(1000 / max(1.0, self.fps * self.play_speed)))
        self._play_after_id = self.root.after(delay_ms, self._play_tick)

    def _jump_start(self):
        self._stop_playback()
        self.scrub.set(int(self.start_var.get()))
        self._show_frame(int(self.scrub.get()))

    def _jump_end(self):
        self._stop_playback()
        self.scrub.set(int(self.end_var.get()))
        self._show_frame(int(self.scrub.get()))

    # ── timeline

    def _frame_to_x(self, f: int) -> int:
        w = max(1, self.timeline.winfo_width())
        return int((f / max(1, self.n_frames - 1)) * (w - 1))

    def _x_to_frame(self, x: int) -> int:
        w = max(1, self.timeline.winfo_width())
        pct = min(1.0, max(0.0, x / max(1, w - 1)))
        return int(round(pct * (self.n_frames - 1)))

    def _draw_timeline(self):
        c = self.timeline
        c.delete("all")
        w = max(1, c.winfo_width()); h = TIMELINE_H
        # background
        c.create_rectangle(0, 0, w, h, fill="#f5f7fb", outline="")
        # ROI
        s = int(self.start_var.get()); e = int(self.end_var.get())
        xs, xe = self._frame_to_x(s), self._frame_to_x(e)
        c.create_rectangle(xs, 0, xe, h, fill="#cfe9ff", outline="")
        # handles
        c.create_rectangle(xs - HANDLE_W // 2, 0, xs + HANDLE_W // 2, h, fill="#2b6cb0", outline="")
        c.create_rectangle(xe - HANDLE_W // 2, 0, xe + HANDLE_W // 2, h, fill="#2b6cb0", outline="")
        # playhead
        xph = self._frame_to_x(int(self.scrub.get()))
        c.create_line(xph, 0, xph, h, fill="#d00", width=2)

    def _timeline_click(self, event):
        x = event.x
        xs = self._frame_to_x(int(self.start_var.get()))
        xe = self._frame_to_x(int(self.end_var.get()))
        if abs(x - xs) <= HANDLE_W:
            self._drag_target = "start"
        elif abs(x - xe) <= HANDLE_W:
            self._drag_target = "end"
        else:
            self.scrub.set(self._x_to_frame(x))
            self._show_frame(int(self.scrub.get()))
            self._drag_target = None

    def _timeline_drag(self, event):
        if not self._drag_target:
            return
        f = self._x_to_frame(event.x)
        if self._drag_target == "start":
            f = min(f, int(self.end_var.get()) - 1)
            f = max(0, f)
            self.start_var.set(f)
        else:
            f = max(f, int(self.start_var.get()) + 1)
            f = min(self.n_frames - 1, f)
            self.end_var.set(f)
        self._draw_timeline()
        self._update_status()

    def _timeline_release(self, _event):
        self._drag_target = None

    # ── actions

    def _set_start_current(self):
        f = int(self.scrub.get())
        if f >= int(self.end_var.get()):
            self.end_var.set(min(self.n_frames - 1, f + 1))
        self.start_var.set(f)
        self._draw_timeline()
        self._update_status()

    def _set_end_current(self):
        f = int(self.scrub.get())
        if f <= int(self.start_var.get()):
            self.start_var.set(max(0, f - 1))
        self.end_var.set(f)
        self._draw_timeline()
        self._update_status()

    def _select_roi(self):
        idx = int(self.scrub.get())
        ok, frame = self._read_frame(idx)
        if not ok:
            return
        r = cv2.selectROI("Select ROI (ENTER to confirm)", frame, fromCenter=False, showCrosshair=True)
        cv2.destroyWindow("Select ROI (ENTER to confirm)")
        if r is None:
            return
        x, y, w, h = map(int, r)
        if w > 0 and h > 0:
            self.roi = (x, y, w, h)
            messagebox.showinfo("ROI selected", f"x={x}, y={y}, w={w}, h={h}")
        self._update_status()

    def _clear_roi(self):
        self.roi = None
        self._update_status()

    def _save_clip(self):
        s = int(self.start_var.get()); e = int(self.end_var.get())
        if e <= s:
            messagebox.showerror("Invalid range", "End must be greater than start.")
            return
        src = self.videos[self.idx]
        dst = self.out_dir
        safe_mkdir(dst)
        try:
            out = write_clip(src, dst, s, e, self.roi)
        except Exception as ex:
            messagebox.showerror("Error", str(ex))
            return
        messagebox.showinfo("Saved", f"Saved: {out}")
        self._refresh_row(self.idx)

    # ── dirs / status

    def _choose_folder(self):
        new = filedialog.askdirectory(initialdir=str(self.video_dir))
        if new:
            self.video_dir = Path(new)
            self.videos = list_videos(self.video_dir)
            self._populate_tree()
            self._select_row_by_index(0)
            self.path_label.config(text=f"Videos: {self.video_dir}   |   Output: {self.out_dir}")

    def _choose_out(self):
        new = filedialog.askdirectory(initialdir=str(self.out_dir))
        if new:
            self.out_dir = Path(new)
            self._populate_tree()
            self._select_row_by_index(self.idx)
            self.path_label.config(text=f"Videos: {self.video_dir}   |   Output: {self.out_dir}")

    def _refresh_row(self, i: int):
        p = self.videos[i]
        try:
            fps, n, _w, _h = probe_video(p)
            dur = n / fps if fps > 0 else 0.0
        except Exception:
            fps, n, dur = 0.0, 0, 0.0
        clips = list_clips_for(p.stem, self.out_dir)
        tag = "done" if clips else "todo"
        self.tree.item(str(i), values=(p.name, f"{fps:.1f}", f"{n}", f"{dur:.2f}", str(len(clips))), tags=(tag,))
        self._update_status()

    def _update_status(self):
        p = self.videos[self.idx]
        clips = list_clips_for(p.stem, self.out_dir)
        roi_txt = "ROI: none" if self.roi is None else f"ROI: {self.roi}"
        self.status.config(text=f"{p.name}  |  saved clips: {len(clips)}  |  start={self.start_var.get()}  end={self.end_var.get()}  |  {roi_txt}")

# ───────── entrypoint ─────────

def main():
    parser = argparse.ArgumentParser(description="Twister Clip Selector")
    parser.add_argument("--videos", type=Path, default=Path("./data"), help="Root folder with videos")
    parser.add_argument("--out",    type=Path, default=Path("./clips"), help="Output folder for clips")
    args = parser.parse_args()
    app = ClipGUI(args.videos, args.out)
    try:
        app.root.mainloop()
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

if __name__ == "__main__":
    main()
