import os
import cv2
import numpy as np

def extract_video_details(video_path):
    """Extract video details (width, height, frame count, duration, fps) using OpenCV."""
    
    # if video_path is provided as a list of parts, join them
    if isinstance(video_path, list):
        video_path = os.path.join(*video_path)
        
    # open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # read properties
    width     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps       = cap.get(cv2.CAP_PROP_FPS)
    n_frames  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # compute duration; guard against fps=0
    if fps > 0:
        duration = n_frames / fps
    else:
        duration = float('nan')
    
    cap.release()
    
    return {
        'width':    width,
        'height':   height,
        'n_frames': n_frames,
        'duration': np.float64(duration),
        'fps':      fps,
        'path':     video_path
    }
