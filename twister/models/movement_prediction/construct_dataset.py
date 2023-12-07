import os
import random
from random import shuffle

import imutils
import pandas as pd

from twister.videos.preprocess import get_video_filenames, preprocess_videos
from twister.videos.utils import extract_video_details
from twister.models.utils import construct_movement_labels, create_dataset,  construct_data_dictionary

from poser.patients import Patient, PatientCollection

# define folders with data
video_directory = '/media/robert/Extreme SSD/ResearchProjects/twister_project/data/raw_data/Healthy_controls/videos/'
labels_directory = '/media/robert/Extreme SSD/ResearchProjects/twister_project/data/raw_data/groundtruth_labels/movement_labels/'
model_directory = '/media/robert/Extreme SSD/ResearchProjects/twister_project/data/model_datasets/movement_model_dataset/'

#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                     Define set of videos to create dataset                 *#
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#


# extract video filenames which we want to use
videos = get_video_filenames(video_directory)

# preprocess videos if not already done
videos = preprocess_videos(videos)


#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                         Create patient objects                             *#
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#


# defining movement labels
movement_labels = pd.read_csv(labels_directory +'movement_labels.csv',header=[0])

# create patient objects for each video 
patients = []
for video_path in videos:
    
    # extract patient id
    patient_id = os.path.basename(video_path).split('_')[0]

    # extract movement times
    movement_times = pd.read_csv(labels_directory +'labels_' + patient_id + '.csv', sep=None, engine='python', header=[0])
        
    # extract video details
    video_details = extract_video_details(video_path)    
    
    # create patient object
    p = Patient(sampling_frequency=video_details['fps'],
                patient_id=patient_id,
                video_details = video_details
                )
    
    # append movement times and labels
    p.movement_times = movement_times
    
    # append to patient list
    patients.append(p)
    

# construct patient objects
pc = PatientCollection()
pc.add_patient_list(patients)

# set buffer of 0.5 seconds for defining movements
buffer = 0.5

# loop over patients label each frame
for patient in pc.patients:
    patient = construct_movement_labels(patient, movement_labels, use_extent=False, buffer=buffer)    


#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                split data into training and validation                     *#
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#


# shuffle patient ids
patient_ids = pc.get_patient_ids()  
random.seed(1)      
shuffle(patient_ids)

# defining training and validation patients
train_patients = patient_ids[:16]
val_patients = patient_ids[16:]


#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#
#*                Construct dataset for model training                        *#
#* -------------------------------------------------------------------------- *#
#* -------------------------------------------------------------------------- *#


n_frames = 10 # number of frames from each patient
label_id ='movement' # we want the frame labels with column movement
label_dict = movement_labels  # using the movement labels as the label dictionary

# define directories for storing data for model
train_directory = model_directory + 'train_data/'
val_directory = model_directory + 'val_data/'

# construct training dataset
create_dataset(pc, label_id, label_dict, n_frames, train_directory, patient_ids=train_patients)  
create_dataset(pc, label_id, label_dict, n_frames, val_directory, patient_ids=val_patients)  

# save label dictionary to folder
movement_labels.to_csv(model_directory +'label_dict.csv', sep=',')

# construct data dictionaries for reading from pytorch
train_data = construct_data_dictionary(train_directory, movement_labels, label_id)
val_data = construct_data_dictionary(val_directory, movement_labels, label_id)
