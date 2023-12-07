import os

import ffmpeg
import numpy as np

def extract_video_details(video_path):
    """ extract video details given a path """
    
    # if filename is combination of root and filename then combine
    if isinstance(video_path, list):
        video_path = os.path.join(*video_path)
        
    # empty dictionary to fill with info
    video_details={}
    
    # probe with ffmpeg
    probe = ffmpeg.probe(video_path)
    
    # stream
    video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    
    # extract info
    video_details['width'] = int(video_stream['width'])
    video_details['height'] = int(video_stream['height'])    
    video_details['n_frames'] = int(video_stream['nb_frames'])    
    video_details['duration'] = np.float64(video_stream['duration'])
    video_details['fps'] = video_details['n_frames']/video_details['duration']   
    video_details['path'] = video_path
    
    return video_details
