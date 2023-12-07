
import numpy as np
import pandas as pd

import skvideo.io
import skvideo.datasets
from tqdm import tqdm

import skvideo
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from twister.videos.preprocess import get_video_filenames
from twister.videos.utils import extract_video_details

from twister.models.cnn_model import load_trained_model
from twister.models.predictions import predict_single_video

# defining folders
model_directory = '/media/robert/Extreme SSD/ResearchProjects/twister_project/data/models/movement_models/'    
video_directory = '/media/robert/Extreme SSD/ResearchProjects/twister_project/data/raw_data/movement_annotated_patients/videos/'
output_directory ='/media/robert/Extreme SSD/ResearchProjects/twister_project/data/predictions/movement_predictions/example_videos/'
data_directory = '/media/robert/Extreme SSD/ResearchProjects/twister_project/data/model_datasets/movement_model_dataset/'

# getting label dictionary
movement_labels = pd.read_csv(data_directory +'label_dict.csv', index_col = 0)

# get videos in directory
videos = get_video_filenames(video_directory)

# load prediction model
n_output = 7
model_path = model_directory + 'model_multilabel.pth'
cnn = load_trained_model(model_path, n_output)

for video in videos:
    
    # extract video details
    video_details = extract_video_details(video)
    
    # predict video with trained model 
    predictions, probabilities, outputs = predict_single_video(video_details, cnn)
    
    # converting into numpy array
    predictions = np.vstack(predictions)  

    
    # get patient id    
    patient_id = video[1].split('.')[0]
        
    # defining font for annotating video
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",20, encoding="unic")
    
    # constructing writer for video
    writer = skvideo.io.FFmpegWriter("{}{}_score_annotated.MOV".format(output_directory, patient_id))

    # construct video generator  
    videogen = skvideo.io.vreader(video_details['path'])    
    
    for i, frame in tqdm(enumerate(videogen), total=video_details['n_frames']):    
        
        # extract frame
        f = Image.fromarray(frame)
        draw = ImageDraw.Draw(f)       
            
        # getting predictions > 0.5 
        label_id = np.where(predictions[i,:] > 0.5)[0].squeeze()
        label = movement_labels.loc[label_id].movement
        
        if isinstance(label,pd.Series):
            label = ' + '.join(label)
        
        # draw label as text in red
        draw.text((5, 5),label,'red',font=font)
        f = np.asarray(f).copy()
        f.setflags(write=1)
        
        # write frame to video
        writer.writeFrame(f)
    
    # close writer 
    writer.close()
























