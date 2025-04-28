import itertools
import os
import importlib
import pkg_resources
import pickle

from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from twister.models.cnn_model import predict_label, load_trained_model
from twister.models.mediapipe_landmarks import prepare_empty_dataframe
from twister.io import save_dataset, save_csv

from procrustes import rotational

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions


def predict_patients(tw,
                     save_temp_object=False,
                     save_temp_csv=False,
                     save_tracking=False,
                     make_video=False,
                     video_folder='./tracking/',
                     csv_folder='./csv_predictions/',
                     recompute_existing=True):
    
    patient_collection = tw.patient_collection
    model_details = tw.model_details
    
    # loop over each model
    for model in model_details: 
           
        if model == 'mediapipe':
            
            # loop over each patient
            for i, patient in enumerate(patient_collection):  
                
                # create empty list to store predictions for each video
                patient.twister_predictions[model] = []              
                
                for v, video in enumerate(patient.video_details):
                    
                    filename = patient.patient_id + '_mediapipe_predictions_v{}.csv'.format(v)
                    if not os.path.isfile(os.path.join(csv_folder, filename)) or recompute_existing:
                        results = {}
    
                        predictions = predict_single_video_mediapipe(video, make_video=make_video, video_folder=video_folder)
    
                        # store all outputs in dictionary
                        results['predictions'] = predictions
                        
                        # add predictions
                        patient.twister_predictions[model].append(results)
                        
                        # save files temporarily in case of crash
                        if save_temp_csv:
                            save_csv(predictions,
                                     patient.patient_id + '_mediapipe_predictions_v{}.csv'.format(v),
                                     folder=csv_folder)
                        elif save_temp_object:
                            save_dataset(tw, 'temp', folder='./')
                    else:
                        print('This video has already been tracked and output as a csv')
    
        # otherwise use standard cnn models
        else:
        
            # load trained model
            model_path = pkg_resources.resource_filename('twister.models.movement_models', 'model_multilabel.pth')
            cnn = load_trained_model(model_path, model_details[model]['n_output'])
    
            # loop over each patient
            for patient in patient_collection:  
                
                if not patient.twister_predictions.get(model, []):
                    # create empty list to store predictions for each video
                    patient.twister_predictions[model] = []
          
                    for v, video in enumerate(patient.video_details):
                        
                        filename = patient.patient_id + '_movement_outputs_v{}.csv'.format(v)
                        if not os.path.isfile(os.path.join(csv_folder, filename)) or recompute_existing:
                        
                            # create empty dict for storing results
                            results = {}    
                            
                            # predict video with trained model
                            predictions, probabilities, outputs = predict_single_video(video, cnn)
                        
                            # store all outputs in dictionary
                            results['predictions'] = pd.DataFrame(predictions, columns=model_details[model]['label_names'])
                            results['probabilities'] = pd.DataFrame(probabilities, columns=model_details[model]['label_names'])
                            results['outputs'] = pd.DataFrame(outputs, columns=model_details[model]['label_names'])
                            
                            patient.twister_predictions[model].append(results)
                            
                            # save files temporarily in case of crash
                            if save_temp_csv:
                                save_csv(results['predictions'],
                                         patient.patient_id + '_movement_predictions_v{}.csv'.format(v),
                                         folder=csv_folder)
                                save_csv(results['probabilities'],
                                         patient.patient_id + '_movement_probabilities_v{}.csv'.format(v),
                                         folder=csv_folder)
                                save_csv(results['outputs'],
                                         patient.patient_id + '_movement_outputs_v{}.csv'.format(v),
                                         folder=csv_folder)
                            elif save_temp_object:
                                save_dataset(tw, 'temp', folder='./')
                        else:
                            print('This video has already been tracked and output as a csv')
    
    # saving object at end
    save_dataset(tw, 'temp', folder='./')
    
    return patient_collection


def predict_single_video(video_details, cnn_model):
    """Predict score for a single video using a CNN model."""
    
    video_path = video_details['path']
    n_frames = video_details['n_frames']
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None, None, None

    predictions = []
    probabilities = []
    outputs = []
    
    for i in tqdm(range(n_frames), total=n_frames, desc="Processing video with CNN"):
        ret, frame = cap.read()
        if not ret:
            break
        out, probs, pred = predict_label(frame, cnn_model)
        predictions.append(pred)
        probabilities.append(probs)
        outputs.append(out)
        
    cap.release()
    return predictions, probabilities, outputs


def predict_single_video_mediapipe(video_details, make_video=False, video_folder='./tracking/', plot=False):
    """Predict score for a single video using Mediapipe."""
    
    # load the faceforward mesh
    face_forward = pickle.load(importlib.resources.open_binary("twister.models", "average_face_mask.pkl"))    

    # load models for tracking with Mediapipe
    face_mesh, pose, hands = load_mediapipe_models()
    
    # get video metadata
    video_path = video_details['path']
    n_frames = video_details['n_frames']
    fps = int(video_details['fps'])
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    
    # Read the first frame to obtain frame dimensions.
    ret, frame = cap.read()
    if not ret:
        print("No frames found in video.")
        cap.release()
        return None
    
    # If making an output video, initialize writer using first frame dimensions.
    if make_video:
        if not os.path.exists(video_folder):
            os.makedirs(video_folder)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = os.path.join(video_folder, video_name + '_MPtracked.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width, _ = frame.shape
        writer = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
    
    # Construct empty dataframes for tracking predictions.
    angle_predictions = pd.DataFrame(index=range(n_frames), columns=['anteroretrocollis','torticollis','laterocollis','shoulder_angle'])
    blend_predictions = pd.DataFrame(index=range(n_frames))
    
    # If the first frame has been read already, process it and then continue.
    frame_idx = 0
    pbar = tqdm(total=n_frames, desc="Processing video with Mediapipe")
    
    # Process the first frame.
    while ret and frame_idx < n_frames:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Note: Original code computes frame_ms as fps * i.
        frame_ms = fps * frame_idx
        
        results_face = face_mesh.detect_for_video(mp_image, frame_ms)
        results_pose = pose.detect_for_video(mp_image, frame_ms)
        results_hands = hands.detect_for_video(mp_image, frame_ms)
        
        # Process pose landmarks.
        if results_pose.pose_world_landmarks:
            pose_df, pose_mapping = prepare_empty_dataframe(hands=True, pose=True, face_mesh=False)
            for l, landmark in enumerate(results_pose.pose_world_landmarks[0]):
                marker = pose_mapping['pose'][l]
                pose_df.loc[frame_idx, (marker, 'x')] = landmark.x
                pose_df.loc[frame_idx, (marker, 'y')] = landmark.y
                pose_df.loc[frame_idx, (marker, 'z')] = landmark.z
                pose_df.loc[frame_idx, (marker, 'visibility')] = landmark.visibility
                pose_df.loc[frame_idx, (marker, 'presence')] = landmark.presence
        
        # Process face landmarks.
        if results_face.face_landmarks:
            face_3d = []
            for idx, landmark in enumerate(results_face.face_landmarks[0]):
                face_3d.append([landmark.x, landmark.y, landmark.z])
            face_3d = np.array(face_3d, dtype=np.float64)
            euler_angles, shoulder_angle = get_head_angle(pose_df, face_forward, face_3d)            
        else: 
            euler_angles =  [np.nan, np.nan, np.nan]    
            shoulder_angle = np.nan
        
        angle_predictions.loc[frame_idx, ['anteroretrocollis','torticollis','laterocollis']] = euler_angles
        angle_predictions.loc[frame_idx, 'shoulder_angle'] = shoulder_angle
        
        if plot and (frame_idx % 5 == 0):
            annotated_image = draw_face_landmarks_on_image(frame, results_face)
            annotated_image = draw_pose_landmarks_on_image(annotated_image, results_pose)
            plt.figure()
            plt.imshow(annotated_image)
            plt.title(str(euler_angles))
        
        if results_face.face_blendshapes:
            for bl in results_face.face_blendshapes[0]:
                blend_predictions.loc[frame_idx, bl.category_name] = bl.score
                
        # If making an output video, annotate the frame and write it.
        if make_video:
            annotated_image = frame.copy()
            if results_face.face_landmarks:
                annotated_image = draw_face_landmarks_on_image(annotated_image, results_face)
            if results_pose.pose_world_landmarks:
                annotated_image = draw_pose_landmarks_on_image(annotated_image, results_pose)
            if results_hands.hand_landmarks:
                annotated_image = draw_hand_landmarks_on_image(annotated_image, results_hands)
            writer.write(annotated_image)
        
        frame_idx += 1
        pbar.update(1)
        ret, frame = cap.read()
    
    pbar.close()
    cap.release()
    if make_video:
        writer.release()
    
    # Combine predictions into a single dataframe.
    predictions = pd.concat([angle_predictions.astype(float), blend_predictions], axis=1)
    return predictions


def get_head_angle(pose_df, face_forward, face_3d):
    """Find head angle relative to face forward."""
    
    # get the angle of the shoulders to the horizontal
    delta_x = (pose_df[('left_shoulder','x')] - pose_df[('right_shoulder','x')]).astype(float).values[0]
    delta_y = (pose_df[('left_shoulder','y')] - pose_df[('right_shoulder','y')]).astype(float).values[0]
    angle = -np.arctan2(delta_y, delta_x)
    
    # Rotation matrix
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    rotation_matrix_2d = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    
    # rotate the face forward mesh
    rotated_face_forward = np.dot(face_forward[:, :2], rotation_matrix_2d)
    rotated_face_forward = np.hstack([rotated_face_forward, face_forward[:, 2].reshape(-1, 1)])
    
    # compute angle of real mesh to new faceforward
    result = rotational(rotated_face_forward, face_3d, scale=True, translate=True)
    euler_angles = rotation_matrix_to_euler_angles(result.t)
    
    return euler_angles, np.rad2deg(angle)


def rot2eul(R):
    beta = -np.arcsin(R[2, 0])  # rotation (head left is positive)
    alpha = np.arctan2(R[2, 1] / np.cos(beta), R[2, 2] / np.cos(beta))
    gamma = np.arctan2(R[1, 0] / np.cos(beta), R[0, 0] / np.cos(beta))
    return np.array((alpha, beta, gamma))


def rotation_matrix_to_euler_angles(R):
    epsilon = 1e-6

    if abs(R[2, 0]) != 1:
        theta1 = -np.arcsin(R[2, 0])
        theta2 = np.pi - theta1

        psi1 = np.arctan2(R[2, 1] / np.cos(theta1), R[2, 2] / np.cos(theta1))
        psi2 = np.arctan2(R[2, 1] / np.cos(theta2), R[2, 2] / np.cos(theta2))

        phi1 = np.arctan2(R[1, 0] / np.cos(theta1), R[0, 0] / np.cos(theta1))
        phi2 = np.arctan2(R[1, 0] / np.cos(theta2), R[0, 0] / np.cos(theta2))

        A = np.array([psi1, theta1, phi1]) * 180 / np.pi
        B = np.array([psi2, theta2, phi2]) * 180 / np.pi
        
        solutions = [A, B]        
        solution = solutions[np.argmin(np.abs(np.sum(solutions, axis=1)))]
        
        return solution
    else:
        phi = 0
        if R[2, 0] == -1:
            theta = np.pi / 2
            psi = phi + np.arctan2(R[1, 2], R[1, 1])
        else:
            theta = -np.pi / 2
            psi = -phi + np.arctan2(-R[1, 2], -R[1, 1])
        return np.array([psi, theta, phi]) * 180 / np.pi


def load_mediapipe_models():
    
    HAND_MODEL = pkg_resources.resource_filename('twister.models.mediapipe_models', 'hand_landmarker.task')
    VisionRunningMode = mp.tasks.vision.RunningMode
    
    # Hand detection
    base_options = mp.tasks.BaseOptions(model_asset_path=HAND_MODEL)
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    options = HandLandmarkerOptions(base_options=base_options,
                                    num_hands=2,
                                    min_hand_detection_confidence=0.1,
                                    min_hand_presence_confidence=0.1,
                                    min_tracking_confidence=0.1,
                                    running_mode=VisionRunningMode.VIDEO)
    
    hands = HandLandmarker.create_from_options(options)
    
    FACE_MODEL = pkg_resources.resource_filename('twister.models.mediapipe_models', 'face_landmarker.task')
    VisionRunningMode = mp.tasks.vision.RunningMode
    base_options = mp.tasks.BaseOptions(model_asset_path=FACE_MODEL)
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
           
    options = FaceLandmarkerOptions(base_options=base_options,
                                    num_faces=1,
                                    min_face_detection_confidence=0,
                                    min_face_presence_confidence=0,
                                    min_tracking_confidence=0,
                                    output_face_blendshapes=True,
                                    running_mode=VisionRunningMode.VIDEO)
    
    face = FaceLandmarker.create_from_options(options)
    
    POSE_MODEL = pkg_resources.resource_filename('twister.models.mediapipe_models', 'pose_landmarker_heavy.task')
    base_options = mp.tasks.BaseOptions(model_asset_path=POSE_MODEL)
    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    options = PoseLandmarkerOptions(base_options=base_options,
                                    running_mode=VisionRunningMode.VIDEO)
  
    pose = PoseLandmarker.create_from_options(options)    
    
    return face, pose, hands


def draw_face_landmarks_on_image(rgb_image, detection_result):
    face_landmarks_list = detection_result.face_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through each detected face.
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
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
        
        solutions.drawing_utils.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks_proto,
            connections=mp.solutions.face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())
    
    return annotated_image


def draw_pose_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)
    
    # Loop through each detected pose.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        
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
    
    # Loop through each detected hand.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        
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
