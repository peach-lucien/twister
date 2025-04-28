import os
import cv2
import numpy as np
from tqdm import tqdm

def get_video_filenames(directory):
    """ Get all video files in `directory` (recursively). """
    filetypes = ('.MOV', '.mov', '.mp4')
    video_files = []
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.endswith(filetypes) and 'preprocessed' not in fname:
                video_files.append([root, fname])
    return video_files

def check_folder(directory):
    """ Create `directory` if it doesn’t exist. """
    if not os.path.exists(directory):
        os.makedirs(directory)

def preprocess_videos(videos):
    """
    Convert each video to a smaller/compressed format.

    - Reads with OpenCV
    - Writes with OpenCV using MPEG-4 (‘mp4v’) codec at the original size/fps
    - Skips already-processed files
    """
    output_filenames = []

    for video_path in videos:
        # support ['root', 'file'] lists
        if isinstance(video_path, list):
            video_path = os.path.join(*video_path)

        # prepare output path
        output_folder   = os.path.join(os.path.dirname(video_path), 'preprocessed_videos')
        base, _ext      = os.path.splitext(os.path.basename(video_path))
        output_filename = os.path.join(output_folder, f"{base}_preprocessed.mov")
        output_filenames.append(output_filename)

        # skip if already done
        if os.path.isfile(output_filename):
            print(f"[skipping] {output_filename} already exists")
            continue

        if 'preprocessed' in os.path.basename(video_path):
            continue

        check_folder(output_folder)

        # open input
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[error] cannot open {video_path}")
            continue

        # grab properties
        fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # set up writer (MPEG-4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        if not writer.isOpened():
            print(f"[error] cannot write to {output_filename}")
            cap.release()
            continue

        # process frames
        for _ in tqdm(range(n_frames), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break

            # — here you could resize or crop `frame` if needed —
            writer.write(frame)

        cap.release()
        writer.release()

    return output_filenames
