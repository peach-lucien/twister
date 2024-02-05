import os
import pickle

import pandas as pd
import re

from twister.data.patients import PatientCollection, Patient
from twister.videos.utils import extract_video_details



def save_csv(df, filename, folder="./datasets"):
    """Save a dataset in a pickle."""
    
    if not os.path.exists(folder):
        os.mkdir(folder)

    df.to_csv(folder + filename)
    
def load_from_csv(folder='./csv_predictions/'):
    """ load from the saved csv files """
    
    
    csvs = os.listdir(folder)
    
    patient_ids = list(set([re.split("_mediapipe|_movement", u)[0] for u in csvs]))
    
    # empty patient list
    patients = []
    for patient_id in patient_ids:
    
        # construct patient object
        p = Patient(
                    patient_id=patient_id,
                    )
    
        # empty dictionary for prediction results
        p.twister_predictions = {'movement':[],
                                 'mediapipe':[]}   
        patients.append(p)
       
    # create empty patient collection
    pc = PatientCollection()
    
    # load patients into patient collection
    pc.add_patient_list(patients) 
    
    # load csvs into patients
    for csv in csvs:
        # extract information about tracking from csv file
        patient_id = re.split("_mediapipe|_movement", csv)[0] 
        model_out = re.split(patient_id+'_', csv)[1].split('_')[0]
        model_type = re.split(patient_id+'_', csv)[1].split('_')[1]
        
        # find the relevant patient
        p = pc.get_patient(patient_id)
        
        if not p.twister_predictions[model_out]:
            results = {model_type:pd.read_csv(folder + csv,index_col=0)}
            p.twister_predictions[model_out].append(results)

        else:
            p.twister_predictions[model_out][0][model_type] = pd.read_csv(folder + csv,index_col=0)

        
        # load the csv file of tracking and save in dictionary
        
        # append tracked video results to patient 
        #p.twister_predictions[model_out].append(results)
        
    
    return pc
    

def save_dataset(obj, filename, folder="./datasets"):
    """Save a dataset in a pickle."""
    
    if not os.path.exists(folder):
        os.mkdir(folder)

    with open(os.path.join(folder, filename + ".pkl"), "wb") as f:
        pickle.dump(obj, f)


def load_dataset(filename):
    """Load a dataset from a pickle."""
    with open(filename, "rb") as f:
        return pickle.load(f)


def construct_patient_collection(videos, patient_ids=None):
    """ construct patient collection using poser """

    # empty patient list
    patients = []
    
    for i, video in enumerate(videos):
        # construct patient object
        if not patient_ids:
            patient = construct_patient(video)
        else:
            patient = construct_patient(video,patient_id=patient_ids[i])
        
        patients.append(patient)
       
    # create empty patient collection
    pc = PatientCollection()
    
    # load patients into patient collection
    pc.add_patient_list(patients)    
    
    return pc

def construct_patient(video, patient_id=None):
    """ construct a single patient """    

    # extract video file name if not given
    if not patient_id:
        if isinstance(video, list):
            patient_id = os.path.basename(video[0]).split('_preprocessed')[0]
        else:
            patient_id = os.path.basename(video).split('_preprocessed')[0]
    
    # extract video details
    if isinstance(video, list):
        video_details = []
        for v in video:
            video_details.append(extract_video_details(v))    
    else:
        video_details = [extract_video_details(video)]
    
    # create patient object
    p = Patient(sampling_frequency=video_details[0]['fps'],
                patient_id=patient_id,
                video_details = video_details
                )

    # empty dictionary for prediction results
    p.twister_predictions = {'movement':None,
                             'mediapipe':None}    

    return p