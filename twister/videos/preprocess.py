
import numpy as np
from skvideo import io
from tqdm import tqdm
import os

def get_video_filenames(directory):
    """ get all videos in directory """
    
    # define video types to search for
    filetypes = ('.MOV','.mov','.mp4')
    video_files = []
    
    # search across all sub directories
    for root, dirs, files in os.walk(directory):        
        for file in files:                  
            # only chosen file extensions and not already preprocessed videos
            if file.endswith(filetypes) and 'preprocessed' not in file:                
                video_files.append([root, file])  
                
    return video_files        
   
def check_folder(directory):
    """ make directory if not exist """
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def preprocess_videos(videos):
    """ converting videos to smaller format and cropping """
    
    # collect of filenames of processed videos
    output_filenames = []
    
    # loop over each video filename
    for video_path in videos:
        
        # if filename is combination of root and filename then combine
        if isinstance(video_path, list):
            video_path = os.path.join(*video_path)
        
        # get video path info
        output_folder = os.path.dirname(video_path) + '/preprocessed_videos/'
        filename = os.path.basename(video_path)

        # define output filename
        output_filename = output_folder + filename.split('.')[0] + '_preprocessed.MOV'   
        
        # sort output filename
        output_filenames.append(output_filename)
        
        # if file exists then skip
        if os.path.isfile(output_filename):
            print('Output video already exists - not overwriting.')
            continue 
        
        # if already preprocessed file then skip
        if 'preprocessed' in os.path.split(video_path)[1]:
            continue

        # create folder if not exist
        check_folder(output_folder)        

        # read video
        reader = io.vreader(video_path)  
          
        # extract details
        data = io.ffprobe(video_path)['video']
        rate = data['@r_frame_rate']
        n_frames = np.int(data['@nb_frames'])
        
        # create output 
        output_dict = {'-vcodec': 'libx265',
                       '-r': rate,
                       '-crf': '24',
                       '-vf': "scale=(iw*sar)*max(540.1/(iw*sar)\,540.1/ih):ih*max(540.1/(iw*sar)\,540.1/ih),crop=540:540",
                       }   
        
        # define writer object
        writer = io.FFmpegWriter(output_filename,
                                 outputdict=output_dict,)
        
        # write to output file
        for i, frame in tqdm(enumerate(reader), total=n_frames):     
            writer.writeFrame(frame)
         
        # close writer object
        writer.close()   
        

        
    return output_filenames