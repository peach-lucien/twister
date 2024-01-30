import numpy as np
import pandas as pd

import json
import importlib.resources
import os

from twister.videos.preprocess import get_video_filenames, preprocess_videos
from twister.io import construct_patient_collection
from twister.models.predictions import predict_patients
from twister.statistics.statistics import extract
from twister.plotting.plotting import plot

class twstr:
    """twister standard object class."""

    def __init__(self,
                 video_path=None, 
                 patient_collection=None,
                 patient_ids=None,
                 video_files=None,
                 model_details=None,
                 model_directory=None,
                 plotting_args = {'plotting_folder': './plots/','ext':'.svg'},
                 ):
        
        """init function."""
        class_dir = os.path.dirname(__file__)
        
        self.patient_collection = None
        self.video_path = video_path
        self.video_files = video_files
        self.model_details = model_details
        self.model_directory = os.path.join(class_dir, '/models/mediapipe_models/')
        self.patient_ids = patient_ids
        
        self.plotting_args = plotting_args
        
        if not model_details:
            self.get_model_details_()
        
        
    def run(self, preprocess_videos=True):
        """ main running funciton """
        
        # preprocess the videos in the path
        if preprocess_videos:
            self.preprocess_videos()
        
        # construct patient objects
        self.load_data()
        
        # make deep learning predictions
        self.predict()
        
        # compute statistics for each patient
        self.analyse()
        
        # aggregate feature vectors
        self.aggregate()
        
        # plot a pdf document for the patient
        self.plot()
        
        
    def load_data(self):
        """Load dataset.""" 

        # construct patient collection from videos        
        self.patient_collection = construct_patient_collection(self.video_files, self.patient_ids)
        
       
        
    def preprocess_videos(self):       
        """ preprocessing videos """
        
        # get video filenames from path
        videos = get_video_filenames(self.video_path)
        
        # preprocess videos to reduce size //540 x 540 
        self.video_files = preprocess_videos(videos)



    def predict(self, save=False):  
        """ predicting movements and scores """
        
        # making predictions on each patient
        self.patient_collection = predict_patients(self.patient_collection, self.model_details)
             
   
        
    def analyse(self):  
        """ compute statistics on the movements and scores """   
        
        self.results = extract(self.patient_collection)
        
        
    def plot(self):
        """ plotting results """  
        
        plot(self.results, self.patient_collection, self.plotting_args)
        
        
    def aggregate(self):
        """ aggregate the feature vectors """
        
        all_features = {}
        
        for patient in self.results:
            result = self.results[patient]    
            features = []
            for feature in result:
                features.append(result[feature]['feature_vector'])
            features = pd.concat(features,axis=1)
            all_features[patient] = features
            
        feature_matrix = pd.concat(all_features,ignore_index=True)
        feature_matrix.index = list(all_features.keys())
        
        # convert all features to float
        feature_matrix = feature_matrix.astype(float)
        
        self.feature_matrix = feature_matrix
        
        
       
    def get_model_details_(self):
        """ Loading dictionary of model details """
        with importlib.resources.open_text("twister.models", "model_details.json") as file:
            self.model_details = json.load(file)
            
        if self.model_directory is not None:
            for model in self.model_details:
                self.model_details[model]['model_path'] =  os.path.join(self.model_directory,
                                                                        self.model_details[model]['model_directory'],
                                                                        self.model_details[model]['model_name'])
                
                