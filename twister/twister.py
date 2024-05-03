import numpy as np
import pandas as pd

import json
import importlib.resources
import glob
import os

from twister.videos.preprocess import get_video_filenames, preprocess_videos
from twister.io import construct_patient_collection, load_dataset, find_video_files
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
            
        if self.video_path and not self.video_files:
            self.video_files = find_video_files(self.video_path)
        
        
    def run(self, 
            preprocess_videos=False,
            make_video=False,
            video_folder='./tracking/',
            save_csv=False,
            save_object=True,
            recompute_existing=True,
            ):
        """ main running funciton """
        
        # preprocess the videos in the path
        if preprocess_videos:
            self.preprocess_videos()
        
        # construct patient objects
        self.load_data()
        
        # make deep learning predictions
        self.predict(make_video=make_video,
                     video_folder=video_folder,
                     save_csv=save_csv,
                     save_object=save_object,
                     recompute_existing=recompute_existing,
                     )
        
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



    def predict(self, file=None, 
                save_csv=False,
                save_object=True,
                make_video=False,
                video_folder='./tracking/',
                recompute_existing=True,):        
        """ predicting movements and scores """
                
        if file is not None:
            self = load_dataset(file)
        
        # making predictions on each patient
        self.patient_collection = predict_patients(self, 
                                                   save_temp_csv=save_csv,
                                                   save_temp_object=save_object,
                                                   make_video=make_video,
                                                   video_folder=video_folder,
                                                   recompute_existing=True)
             
   
        
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
                
                