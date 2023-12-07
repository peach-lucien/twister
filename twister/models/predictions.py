
import itertools
import os
import importlib
import pkg_resources

from tqdm import tqdm
import skvideo

import pandas as pd
import numpy as np


# from twister.models.cnn_model import predict_label, load_trained_model

from procrustes import rotational
import mediapipe as mp

import pickle

def predict_patients(patient_collection, model_details):

    # loop over each model
    for model in model_details: 
           
        if model == 'mediapipe':
            
            # loop over each patient
            for patient in patient_collection:          
                
                # create empty list to store predictions for each video
                patient.twister_predictions[model] = []              
                
                for video in patient.video_details:
                    results = {}

                    predictions = predict_single_video_mediapipe( video )

                    # store all outputs in dictionary
                    results['predictions'] = pd.DataFrame(predictions, columns=['anteroretrocollis','rotation','tilt'])
                    
                    patient.twister_predictions[model].append(results)
                    
                    
        # # otherwise use standard cnn models
        # else:
        
        #     # load trained model
        #     cnn = load_trained_model(model_details[model]['model_path'], model_details[model]['n_output'])
    
        #     # loop over each patient
        #     for patient in patient_collection:  
                
        #         # create empty list to store predictions for each video
        #         patient.twister_predictions[model] = []
          
        #         for video in patient.video_details:
        #             # create empty dict for storing results
        #             results = {}

                    
        #             # predict video with trained model
        #             predictions, probabilities, outputs = predict_single_video( video, cnn)
                
        #             # store all outputs in dictionary
        #             results['predictions'] = pd.DataFrame(predictions, columns=model_details[model]['label_names'])
        #             results['probabilities'] = pd.DataFrame(probabilities, columns=model_details[model]['label_names'])
        #             results['outputs'] = pd.DataFrame(outputs, columns=model_details[model]['label_names'])
                    
        #             patient.twister_predictions[model].append(results)
       
    
    return patient_collection


def predict_single_video(video_details, cnn_model):
    """ predict score for a single video """
    
    video_path = video_details['path']
    n_frames = video_details['n_frames']   
     
    videogen = skvideo.io.vreader(video_path)

    predictions = []
    probabilities = []
    outputs=[]
    
    for frame in tqdm( itertools.islice(videogen, n_frames), total=n_frames):     
        out, probs, pred = predict_label(frame, cnn_model)
        predictions.append(pred)
        probabilities.append(probs)
        outputs.append(out)

    return predictions, probabilities, outputs

def predict_single_video_mediapipe(video_details):
    """ predict score for a single video """
    
    #with importlib.resources.read_binary("twister.models", "average_face_mask.pkl") as file:
    # file = importlib.resources.read_binary("twister.models", "average_face_mask.pkl")
    #face_forward = pickle.load(file)
    # with open('average_face_mask.pkl', 'rb') as handle:
    #     face_forward = pickle.load(handle)
    
    
    # load the faceforward mesh
    face_forward = pickle.load(importlib.resources.open_binary("twister.models", "average_face_mask.pkl"))    
    
    
    # mp_face_mesh = mp.solutions.face_mesh
    # face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
    #                                   refine_landmarks=True,
    #                                  min_detection_confidence=0.5,
    #                                  min_tracking_confidence=0.5)
    
    face_mesh, pose = load_mediapipe_models()
    
    video_path = video_details['path']
    n_frames = video_details['n_frames']   
    fps = int(video_details['fps'])
     
    # videogen = skvideo.io.vreader(video_path)
    videogen = list(skvideo.io.vreader(video_path))
    
    
    predictions = []
    #probabilities = []
    #outputs=[]
    
    for i, frame in enumerate(tqdm( itertools.islice(videogen, n_frames), total=n_frames)):     
        
        # converted for MP format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # get frame time in ms
        frame_ms = fps*i
        
        # running mediapipe predictions
        #results_face_mesh = face_mesh.detect_for_video(mp_image,frame_ms)
        
        
        results_face = face_mesh.detect_for_video(mp_image, frame_ms)
        results_pose = pose.detect_for_video(mp_image, frame_ms)
        
        #results = face_mesh.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))        
        # results = face_mesh.process(frame)     
        
        ### get angles of head
        # TODO: make it relative to the shoulders 

        face_3d = []
        if results_face.face_landmarks:
            for face_landmarks in results_face.face_landmarks:
                for idx, lm in enumerate(face_landmarks):                
                    face_3d.append([lm.x,lm.y,lm.z])                        
                face_3d = np.array(face_3d, dtype=np.float64)
                result = rotational(face_forward, face_3d, scale=True, translate=True)                        
                euler_angles = rot2eul(result.t)*180/np.pi
        else:
            euler_angles = [np.nan,np.nan,np.nan]
        
        predictions.append(euler_angles)
        
        
        
        # TODO: use the facial blender predicitons as well

        
        
        #probabilities.append(probs)
        #outputs.append(out)

    return predictions # , probabilities, outputs


def rot2eul(R):
    beta = -np.arcsin(R[2,0])
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta))
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta))
    return np.array((alpha, beta, gamma))



def load_mediapipe_models():
    
    HAND_MODEL = pkg_resources.resource_filename('twister.models.mediapipe_models', 'hand_landmarker.task')
    
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # hand detection
    base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL)
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    options = HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2,
                                    running_mode=VisionRunningMode.VIDEO)
    hands = HandLandmarker.create_from_options(options)
    
    FACE_MODEL = pkg_resources.resource_filename('twister.models.mediapipe_models', 'face_landmarker.task')
    
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # hand detection
    base_options = mp.tasks.BaseOptions(model_asset_path=FACE_MODEL)
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    options = FaceLandmarkerOptions(base_options=base_options,
                                    output_face_blendshapes=True,
                                    running_mode=VisionRunningMode.VIDEO)
    face = FaceLandmarker.create_from_options(options)
    
    POSE_MODEL = pkg_resources.resource_filename('twister.models.mediapipe_models', 'pose_landmarker_heavy.task')
    
    # pose detection
    base_options = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL)
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    options = PoseLandmarkerOptions(base_options=base_options,
                                    running_mode=VisionRunningMode.VIDEO)
    pose = PoseLandmarker.create_from_options(options)    
    
    return face, pose