import os
import pickle

from twister.data.patients import PatientCollection, Patient
from twister.videos.utils import extract_video_details



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
        patient_id = os.path.basename(video[0]).split('_preprocessed')[0]
        
    
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
    p.twister_predictions = {'movement':None}    

    return p