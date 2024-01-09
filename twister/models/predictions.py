
import itertools
import os
import importlib
import pkg_resources

from tqdm import tqdm
import skvideo

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt


from twister.models.cnn_model import predict_label, load_trained_model
from twister.models.mediapipe_landmarks import prepare_empty_dataframe

from procrustes import rotational, orthogonal, generic

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

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
                    results['predictions'] = predictions #pd.DataFrame(predictions, columns=['anteroretrocollis','rotation','tilt'])
                    
                    patient.twister_predictions[model].append(results)
                    
                    
        # otherwise use standard cnn models
        else:
        
            # load trained model
            model_path = pkg_resources.resource_filename('twister.models.movement_models', 'model_multilabel.pth')
            cnn = load_trained_model(model_path, model_details[model]['n_output'])
    
            # loop over each patient
            for patient in patient_collection:  
                
                # create empty list to store predictions for each video
                patient.twister_predictions[model] = []
          
                for video in patient.video_details:
                    # create empty dict for storing results
                    results = {}

                    
                    # predict video with trained model
                    predictions, probabilities, outputs = predict_single_video( video, cnn)
                
                    # store all outputs in dictionary
                    results['predictions'] = pd.DataFrame(predictions, columns=model_details[model]['label_names'])
                    results['probabilities'] = pd.DataFrame(probabilities, columns=model_details[model]['label_names'])
                    results['outputs'] = pd.DataFrame(outputs, columns=model_details[model]['label_names'])
                    
                    patient.twister_predictions[model].append(results)
       
    
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



def predict_single_video_mediapipe(video_details, make_video=False, plot=False):
    """ predict score for a single video """
        
    # load the faceforward mesh
    face_forward = pickle.load(importlib.resources.open_binary("twister.models", "average_face_mask.pkl"))    

    # load models for tracking with mediapipe
    face_mesh, pose = load_mediapipe_models()
    
    # get video metadata
    video_path = video_details['path']
    n_frames = video_details['n_frames']   
    fps = int(video_details['fps'])
     
    # videogen = skvideo.io.vreader(video_path)
    videogen = list(skvideo.io.vreader(video_path))
    
    # defining a video writer
    #if make_video:
    #    writer = skvideo.io.FFmpegWriter(output_folder + video_name + '_MPtracked.mp4')

    # construct empty dataframe for filling with tracking results    
    
    angle_predictions = pd.DataFrame(index=range(n_frames),columns=['anteroretrocollis','torticollis','laterocollis','shoulder_angle'])
    blend_predictions = pd.DataFrame(index=range(n_frames))
    #probabilities = []
    #outputs=[]
    
    # loop over each frame in the video
    for i, frame in enumerate(tqdm(itertools.islice(videogen, n_frames), total=n_frames)):     
        
        # converted for MP format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        
        # get frame time in ms
        frame_ms = fps*i
        
        # run mediapipe detection
        results_face = face_mesh.detect_for_video(mp_image, frame_ms)
        results_pose = pose.detect_for_video(mp_image, frame_ms)
        
        
        # if the tracking was successful
        if results_pose.pose_world_landmarks:   
            pose_df, pose_mapping = prepare_empty_dataframe(hands=False, pose=True, face_mesh=False)

            # only looping over the first pose (assuming there is only one person in the frame)
            for l, landmark in enumerate(results_pose.pose_world_landmarks[0]):  # TODO use world or not world landmarks
                
                marker = pose_mapping['pose'][l]
                pose_df.loc[i,(marker,'x')] = landmark.x
                pose_df.loc[i,(marker,'y')] = landmark.y
                pose_df.loc[i,(marker,'z')] = landmark.z
                pose_df.loc[i,(marker,'visibility')] = landmark.visibility
                pose_df.loc[i,(marker,'presence')] = landmark.presence


        # extract face mesh with matching markers to pose
        if results_face.face_landmarks:   
            # face_df, face_mapping = prepare_empty_dataframe(hands=False, pose=False, face_mesh=True)
            face_3d = []
            # only looping over the first pose (assuming there is only one person in the frame)
            for idx, landmark in enumerate(results_face.face_landmarks[0]): # l in face_mapping['face'].keys(): 
                
                #landmark = results_face.face_landmarks[0][idx]                
                face_3d.append([landmark.x,landmark.y,landmark.z])                        
                
                # # find mesh face markers that are also in the pose face markers
                # if idx in face_mapping['face'].keys():
                #     marker = face_mapping['face'][idx]
                #     face_df.loc[i,(marker,'x')] = landmark.x
                #     face_df.loc[i,(marker,'y')] = landmark.y
                #     face_df.loc[i,(marker,'z')] = landmark.z
                #     face_df.loc[i,(marker,'visibility')] = landmark.visibility
                #     face_df.loc[i,(marker,'presence')] = landmark.presence  
                
            # this is the coordinates of the fitted face mesh
            face_3d = np.array(face_3d, dtype=np.float64)             

            # find head angles using shoulders as a reference point
            euler_angles, shoulder_angle = get_head_angle(pose_df, face_forward, face_3d)            
        else: 
            euler_angles =  [np.nan,np.nan,np.nan]    
            shoulder_angle = np.nan
        
        # store angle predictions
        #angle_predictions.append(euler_angles)
        angle_predictions.loc[i,['anteroretrocollis','torticollis','laterocollis']] = euler_angles
        angle_predictions.loc[i,'shoulder_angle'] = shoulder_angle
        
        if plot and  (i % 5==0):
            annotated_image = draw_face_landmarks_on_image(frame, results_face)
            annotated_image = draw_pose_landmarks_on_image(annotated_image, results_pose)
            plt.figure();
            plt.imshow(annotated_image);plt.title(euler_angles)        
        
        #plt.figure()
        #plt.scatter(face_forward[:,0], -face_forward[:,1])
        #plt.scatter(rotated_face_forward[:,0], -rotated_face_forward[:,1])
        #plt.scatter(face_3d[:,0], -face_3d[:,1])        
        
        # TODO: use the facial blender predicitons as well
        if results_face.face_blendshapes:   
            for bl in results_face.face_blendshapes[0]:
                blend_predictions.loc[i, bl.category_name] = bl.score                
        
        # annotate frame with tracked markers
        if make_video:
            # creating a copy of image
            annotated_image = frame.copy()  
            
            if results_face.face_landmarks:
                annotated_image = draw_face_landmarks_on_image(annotated_image, results_face)

            if results_pose.pose_world_landmarks:
                annotated_image = draw_pose_landmarks_on_image(annotated_image, results_pose)


    # write annotated image to video
    #if make_video:
        #writer.writeFrame(annotated_image)
    
    # combine predictions into a single dataframe
    predictions = pd.concat([angle_predictions.astype(float),blend_predictions], 1)

    return predictions



def get_head_angle(pose_df, face_forward, face_3d):
    " function to find head angle relative to face forward"    
    
    # get the angle of the shoulders to the horizontal
    delta_x = (pose_df[('left_shoulder','x')] - pose_df[('right_shoulder','x')]).astype(float).values[0]
    delta_y = (pose_df[('left_shoulder','y')] - pose_df[('right_shoulder','y')]).astype(float).values[0]
    angle = -np.arctan2(delta_y, delta_x) # 
    
    # Rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix_2d = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    # rotate the face fortward mesh
    rotated_face_forward = np.dot(face_forward[:,:2], rotation_matrix_2d)
    rotated_face_forward = np.hstack([rotated_face_forward, face_forward[:,2].reshape(-1,1)])
    
    # compute angle of real mesh to new faceforward
    result = rotational(rotated_face_forward, face_3d, scale=True, translate=True)                        
    # euler_angles = rot2eul(result.t)*180/np.pi
    euler_angles = rotation_matrix_to_euler_angles(result.t)   

    return euler_angles, np.rad2deg(angle)   

def rot2eul(R):
    beta = -np.arcsin(R[2,0]) # rotation (head left is positive - from their perspective, and head right is negative)
    alpha = np.arctan2(R[2,1]/np.cos(beta),R[2,2]/np.cos(beta)) # anteroretrocollis (forwards is negative and backwards is positive)
    gamma = np.arctan2(R[1,0]/np.cos(beta),R[0,0]/np.cos(beta)) # tilt (head tilt right is negative and head tilt left is positive)
    return np.array((alpha, beta, gamma))

def rotation_matrix_to_euler_angles(R):
    # Tolerance for singularities detected due to numerical precision
    epsilon = 1e-6

    if abs(R[2, 0]) != 1:
        # R[2, 0] is not ±1
        theta1 = -np.arcsin(R[2, 0])
        theta2 = np.pi - theta1

        psi1 = np.arctan2(R[2, 1] / np.cos(theta1), R[2, 2] / np.cos(theta1))
        psi2 = np.arctan2(R[2, 1] / np.cos(theta2), R[2, 2] / np.cos(theta2))

        phi1 = np.arctan2(R[1, 0] / np.cos(theta1), R[0, 0] / np.cos(theta1))
        phi2 = np.arctan2(R[1, 0] / np.cos(theta2), R[0, 0] / np.cos(theta2))

        # Two possible solutions exist
        A = np.array([psi1, theta1, phi1])*180/np.pi
        B = np.array([psi2, theta2, phi2])*180/np.pi
        
        # find the solution with the lowest possible rotations (most likely scenario)
        solutions = [A,B]        
        solution = solutions[np.argmin(abs(np.sum(solutions,1)))]
        
        return solution
    else:
        # R[2, 0] is ±1
        phi = 0  # Arbitrary value
        if R[2, 0] == -1:
            theta = np.pi / 2
            psi = phi + np.arctan2(R[1, 2], R[1, 1])
        else:
            theta = -np.pi / 2
            psi = -phi + np.arctan2(-R[1, 2], -R[1, 1])

        return np.array([psi, theta, phi])*180/np.pi


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


def draw_face_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected faces to visualize.
    for idx in range(len(face_landmarks_list)):
        face_landmarks = face_landmarks_list[idx]
          
        # Draw the face landmarks.
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
        ])
          
        solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_tesselation_style())
        
        solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles
                .get_default_face_mesh_contours_style())
        
        solutions.drawing_utils.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                  landmark_drawing_spec=None,
                  connection_drawing_spec=mp.solutions.drawing_styles
                  .get_default_face_mesh_iris_connections_style())
    
    return annotated_image


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        
        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          pose_landmarks_proto,
          solutions.pose.POSE_CONNECTIONS,
          solutions.drawing_styles.get_default_pose_landmarks_style())
        
    return annotated_image


def draw_hand_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through the detected poses to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
        # Draw the pose landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style())
        
    return annotated_image