# twister/gui/clip_gui.py
from __future__ import annotations
import argparse
import os
from pathlib import Path
import re
import time
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk

VIDEO_EXTS = (".mp4", ".mov", ".MOV", ".avi", ".mkv")
CLIP_PATTERN = re.compile(r"^(?P<stem>.+)__s(?P<s>\d+)_e(?P<e>\d+)\.mp4$")

TIMELINE_H = 28
HANDLE_W = 8

# ─────────────────────────── filesystem helpers ────────────────────────────

def list_videos(root: Path) -> list[Path]:
    vids = []
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
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n_frames, w, h

def list_clips_for(stem: str, out_dir: Path) -> list[Path]:
    if not out_dir.exists():
        return []
    hits = []
    for p in out_dir.glob(f"{stem}__s*_e*.mp4"):
        if CLIP_PATTERN.match(p.name):
            hits.append(p)
    return sorted(hits)

def write_clip(src: Path, dst_dir: Path, start_f: int, end_f: int,
               roi: tuple[int,int,int,int] | None, fourcc="mp4v") -> Path:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise IOError(f"Cannot open {src}")
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start  = max(0, min(start_f, n_fr-1))
    end    = max(start+1, min(end_f, n_fr))

    if roi:
        x, y, w, h = roi
        w_out, h_out = int(w), int(h)
    else:
        x = y = 0
        w_out, h_out = width, height

    safe_mkdir(dst_dir)
    stem = src.stem
    out = dst_dir / f"{stem}__s{start}_e{end}.mp4"
    fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
    writer = cv2.VideoWriter(str(out), fourcc_code, fps, (w_out, h_out))
    if not writer.isOpened():
        cap.release()
        raise IOError(f"Cannot write to {out}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    for _ in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        if roi:
            frame = frame[y:y+h_out, x:x+w_out]
        writer.write(frame)

    writer.release()
    cap.release()
    return out

# ───────────────────────────── main application ─────────────────────────────

class ClipGUI:
    def __init__(self, video_dir: Path, out_dir: Path):
        self.root = tk.Tk()
        self.root.title("Twister — Clip Selector")
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
        self.out_dir = Path(out_dir)

        self.videos: list[Path] = list_videos(self.video_dir)
        if not self.videos:
            messagebox.showerror("No videos", f"No videos found under {self.video_dir}")
            self.root.destroy()
            return

        # state
        self.idx = 0
        self.cap = None
        self.frame_img = None
        self.roi: tuple[int,int,int,int] | None = None
        self.fps = 30.0
        self.n_frames = 0
        self.width = self.height = 0
        self._playing = False

        # selections
        self.start_var = tk.IntVar(value=0)
        self.end_var   = tk.IntVar(value=0)
        self._drag_target: str | None = None  # "start"|"end"|None

        # build UI
        self._build()
        self._populate_tree()
        self._select_row_by_index(0)

        # shortcuts
        self.root.bind("<space>", lambda e: self._toggle_play())
        self.root.bind("s", lambda e: self._set_start_current())
        self.root.bind("e", lambda e: self._set_end_current())
        self.root.bind("<Left>", lambda e: self._nudge(-1))
        self.root.bind("<Right>", lambda e: self._nudge(1))

    # ───────────────────────────── UI construction ───────────────────────────

    def _build(self):
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", padx=8, pady=6)
        ttk.Button(bar, text="Open Folder…", command=self._choose_folder).pack(side="left")
        ttk.Button(bar, text="Choose Output…", command=self._choose_out).pack(side="left", padx=(6,0))
        self.path_label = ttk.Label(bar, text=f"Videos: {self.video_dir}   |   Output: {self.out_dir}")
        self.path_label.pack(side="left", padx=12)

        paned = ttk.Panedwindow(self.root, orient="horizontal")
        paned.pack(fill="both", expand=True, padx=8, pady=(0,8))

        # left list
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

        # right viewer
        right = ttk.Frame(paned)
        self.title_label = ttk.Label(right, text="—", font=("TkDefaultFont", 12, "bold"))
        self.title_label.pack(anchor="w", pady=(4,2))

        self.canvas = ttk.Label(right)
        self.canvas.pack(pady=4)

        self.info = ttk.Label(right, text="—")
        self.info.pack()

        # playhead scrub (single slider)
        self.scrub = ttk.Scale(right, from_=0, to=100, orient="horizontal", command=self._on_scrub)
        self.scrub.pack(fill="x", padx=4, pady=(6,2))

        # timeline with ROI highlight + handles + playhead
        self.timeline = tk.Canvas(right, height=TIMELINE_H, background="#f5f7fb",
                                  highlightthickness=1, highlightbackground="#ccd")
        self.timeline.pack(fill="x", padx=4, pady=(0,8))
        self.timeline.bind("<Configure>", lambda e: self._draw_timeline())
        self.timeline.bind("<Button-1>", self._timeline_click)
        self.timeline.bind("<B1-Motion>", self._timeline_drag)
        self.timeline.bind("<ButtonRelease-1>", self._timeline_release)

        # actions row
        actions = ttk.Frame(right)
        actions.pack(fill="x", pady=(4,8))
        ttk.Button(actions, text="◀ Prev", command=lambda: self._load_video(self.idx-1)).pack(side="left")
        ttk.Button(actions, text="Next ▶", command=lambda: self._load_video(self.idx+1)).pack(side="left", padx=(6,0))
        ttk.Button(actions, text="Select ROI", command=self._select_roi).pack(side="left", padx=(12,0))
        ttk.Button(actions, text="Clear ROI", command=self._clear_roi).pack(side="left", padx=(6,0))
        ttk.Button(actions, text="Set Start (s)", command=self._set_start_current).pack(side="right", padx=4)
        ttk.Button(actions, text="Set End (e)", command=self._set_end_current).pack(side="right", padx=4)
        ttk.Button(actions, text="Save Clip", command=self._save_clip).pack(side="right", padx=(12,0))

        paned.add(right, weight=3)

        # status bar
        self.status = ttk.Label(self.root, relief="sunken", anchor="w")
        self.status.pack(fill="x", padx=8, pady=(0,8))

    # ───────────────────────────── list / loading ────────────────────────────

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
        idx = int(sel[0])
        self._load_video(idx)

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
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def _load_video(self, i: int):
        if not self.videos:
            return
        self.idx = max(0, min(i, len(self.videos)-1))
        path = self.videos[self.idx]
        if not self._open_cap(path):
            return
        self.title_label.config(text=str(path.name))
        # reset
        self.scrub.configure(to=self.n_frames-1)
        self.start_var.set(0)
        self.end_var.set(self.n_frames-1)
        self.scrub.set(0)
        self.roi = None
        self._show_frame(0)
        self._update_status()
        self._draw_timeline()

    # ───────────────────────────── playback / display ────────────────────────

    def _read_frame(self, idx: int):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        return ok, frame

    def _show_frame(self, idx: int):
        ok, frame = self._read_frame(idx)
        if not ok:
            return
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        max_w = 1024
        if im.width > max_w:
            im = im.resize((max_w, int(im.height * max_w / im.width)))
        self.frame_img = ImageTk.PhotoImage(image=im)
        self.canvas.configure(image=self.frame_img)
        t = idx / self.fps
        self.info.config(text=f"frame {idx}/{self.n_frames-1}  ({t:.2f}s @ {self.fps:.2f} fps)")
        self._draw_timeline()  # update playhead

    def _on_scrub(self, val):
        self._show_frame(int(float(val)))

    def _nudge(self, delta: int):
        new = max(0, min(self.n_frames-1, int(self.scrub.get()) + delta))
        self.scrub.set(new)
        self._show_frame(new)

    def _toggle_play(self):
        self._playing = not self._playing
        last = time.time()
        while self._playing:
            i = int(self.scrub.get())
            if i >= self.n_frames-1:
                self._playing = False
                break
            now = time.time()
            if now - last < 1.0 / max(1.0, self.fps):
                self.root.update()
                continue
            last = now
            self._nudge(+1)
            self.root.update()

    # ───────────────────────────── timeline (single slider UI) ───────────────

    def _frame_to_x(self, f: int) -> int:
        w = max(1, self.timeline.winfo_width())
        return int((f / max(1, self.n_frames-1)) * (w-1))

    def _x_to_frame(self, x: int) -> int:
        w = max(1, self.timeline.winfo_width())
        pct = min(1.0, max(0.0, x / max(1, w-1)))
        return int(round(pct * (self.n_frames-1)))

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
        c.create_rectangle(xs-HANDLE_W//2, 0, xs+HANDLE_W//2, h, fill="#2b6cb0", outline="")
        c.create_rectangle(xe-HANDLE_W//2, 0, xe+HANDLE_W//2, h, fill="#2b6cb0", outline="")
        # playhead
        xph = self._frame_to_x(int(self.scrub.get()))
        c.create_line(xph, 0, xph, h, fill="#d00", width=2)

    def _timeline_click(self, event):
        x = event.x
        xs = self._frame_to_x(int(self.start_var.get()))
        xe = self._frame_to_x(int(self.end_var.get()))
        if abs(x - xs) <= HANDLE_W:  # grab start
            self._drag_target = "start"
        elif abs(x - xe) <= HANDLE_W:  # grab end
            self._drag_target = "end"
        else:
            # jump playhead
            self.scrub.set(self._x_to_frame(x))
            self._show_frame(int(self.scrub.get()))
            self._drag_target = None

    def _timeline_drag(self, event):
        if not self._drag_target:
            return
        f = self._x_to_frame(event.x)
        if self._drag_target == "start":
            f = min(f, int(self.end_var.get())-1)
            f = max(0, f)
            self.start_var.set(f)
        else:
            f = max(f, int(self.start_var.get())+1)
            f = min(self.n_frames-1, f)
            self.end_var.set(f)
        self._draw_timeline()
        self._update_status()

    def _timeline_release(self, _event):
        self._drag_target = None

    # ───────────────────────────── actions ───────────────────────────────────

    def _set_start_current(self):
        f = int(self.scrub.get())
        if f >= int(self.end_var.get()):
            self.end_var.set(min(self.n_frames-1, f+1))
        self.start_var.set(f)
        self._draw_timeline()
        self._update_status()

    def _set_end_current(self):
        f = int(self.scrub.get())
        if f <= int(self.start_var.get()):
            self.start_var.set(max(0, f-1))
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

    # ───────────────────────────── directories / status ──────────────────────

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

# ───────────────────────────── entry point ──────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Twister Clip Selector")
    parser.add_argument("--videos", type=Path, default=Path("./data"), help="Root folder with videos")
    parser.add_argument("--out",    type=Path, default=Path("./clips"), help="Output folder for clips")
    args = parser.parse_args()
    app = ClipGUI(args.videos, args.out)
    try:
        app.root.mainloop()
    finally:
        try: cv2.destroyAllWindows()
        except Exception: pass

if __name__ == "__main__":
    main()
