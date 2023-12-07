import pandas as pd
import numpy as np

import torch

import random
import os

import imageio
from imutils import paths

from tqdm import tqdm
from multiprocessing import Pool

import ffmpeg


def seed_everything(SEED=42):
    ''' SEED everything '''
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = True     
    

def check_folder(directory):
    """ make directory if not exist """
    if not os.path.exists(directory):
        os.makedirs(directory)

class Worker(object):
    """ Frame generation worker for multiprocessing  """
    
    def __init__(self, n_frames, height, width, image_folder, label_id, label_dict):        
        self.n_frames =  n_frames
        self.height = height
        self.width = width
        self.image_folder = image_folder   
        self.label_id = label_id
        self.label_dict = label_dict
        
    def __call__(self, patient):
        frame_extraction(patient, self.n_frames, self.height, self.width, self.image_folder, self.label_id, self.label_dict)
        return 


def get_random_frame_ids(patient, n_frames, label_id):
    """ get frame ids and labels """
    
    # identify frames that have a label
    has_label = ~patient.frame_labels[label_id].isna().all(1)
    
    # check the frame hasn't already been sampled
    not_pre_sampled = patient.frame_labels['pre_sampled']==0
    
    # index of frames that can be sampled for generation
    index_to_sample = patient.frame_labels.index[(has_label & not_pre_sampled)]
    
    # defining random frame ids from the possible frames for sampling
    frame_ids = [random.randint(0, len(index_to_sample)-1) for p in range(n_frames)] 
    frame_index = index_to_sample[frame_ids]
    
    # get the associated label of the frame ids
    labels = patient.frame_labels.loc[frame_index, label_id].values

    # set sampled frames to 'presampled'
    patient.frame_labels.loc[index_to_sample[frame_ids],'pre_sampled'] = 1
    
    return frame_index , labels, patient



def _check_frame_extraction(patients, height, width, image_folder, label_id, label_dict, image_type='png'):
    """ check if frame extraction working """
    
    # just generate one frame from a single patient
    n_frames = 1
    patient = patients[0]
    
    # taking the first frame
    f = 0
    
    # extract random frames
    frame_ids, frame_labels, patient = get_random_frame_ids(patient, n_frames, label_id)
    
    # only use one frame
    frame_num = frame_ids[f]
    
    # load image from video
    video_path = patient.video_details['path']
    frame = load_frame( video_path, frame_num, height=height, width=width)
    
    # extract class label of frame
    if len(label_id)==1:
        # if just a single label
        label = label_dict.loc[int(frame_labels[f]),label_id]
    else:
        # if combining two labels
        label_1 = label_dict[0].loc[int(frame_labels[f,0]),label_id[0]]
        label_2 = label_dict[1].loc[int(frame_labels[f,1]),label_id[1]]
        label = label_1 + '_' + label_2
       
        
    # sub folder for class label
    label_folder = label + '/'
    
    # check directory exists
    check_folder(image_folder + label_folder)
    
    # define path for saving image
    filename = '{}_patient_id_{}_frame_id_{}.{}'.format(label,
                                                         patient.patient_id,
                                                         frame_ids[f],
                                                         image_type)
    
    # write image to file
    imageio.imwrite(image_folder + label_folder + filename , frame)
    
    return


def frame_extraction(patient, n_frames, height, width, image_folder, label_id, label_dict, image_type='png'):
    """ frame extraction for a single patient """
     
    # extract random frames
    frame_ids, frame_labels, patient = get_random_frame_ids(patient, n_frames, label_id)
    
    for f, frame_num in enumerate(frame_ids):    
    
        # load image from video
        video_path = patient.video_details['path']
        frame = load_frame( video_path, frame_num, height=height, width=width)
    
        # extract class label of frame
        if len(label_id)==1:
            # if just a single label
            label = label_dict.loc[int(frame_labels[f]),label_id]
        else:
            # if combining two labels
            label_1 = label_dict[0].loc[int(frame_labels[f,0]),label_id[0]]
            label_2 = label_dict[1].loc[int(frame_labels[f,1]),label_id[1]]
            label = label_1 + '_' + label_2
    
        # sub folder for class label
        label_folder = label + '/'
        
        # check directory exists
        check_folder(image_folder + label_folder)
        
        # define path for saving image
        filename = '{}_patient_id_{}_frame_id_{}.{}'.format(label,
                                                            patient.patient_id,
                                                            frame_ids[f],
                                                            image_type)
        
        # write image to file
        imageio.imwrite(image_folder + label_folder + filename , frame)
    
    return


def load_frame(video_path, frame_num, height=540, width=540):
    """ load frame (fastest direct with ffmpeg) """ 
    
    # use ffmpeg to extract a single frame
    out = (
            ffmpeg
            .input(video_path)
            .filter('select', 'gte(n,{})'.format(frame_num))
            .output('pipe:', vframes=1, format='rawvideo', pix_fmt='rgb24')
            .global_args('-loglevel', 'error')
            .global_args('-y')           
            .run(capture_stdout=True)
            )

    # convert from bytes to numpy array
    frame = np.frombuffer(out[0], np.uint8).reshape([-1, height, width, 3])

    # remove extract dimension and convert to uint8    
    frame = frame.squeeze().astype(np.uint8)

    return frame


    
def create_dataset(patient_collection,
                   label_id,
                   label_dict,
                   n_frames,
                   image_folder,
                   patient_ids=None,
                   height=540,
                   width=540,
                   n_workers=1,
                   run_check=False,):    
    
    # getting list of patients
    if patient_ids is None:
        patients = patient_collection.patients
    else:
        patients = []
        for patient_id in patient_ids: 
            patients.append(patient_collection.get_patient(patient_id))
    
    # checking frame extraction outside of parallel loop
    if run_check:
        _check_frame_extraction(patients, height, width, image_folder, label_id, label_dict)
        return
    
    try:
        worker = Worker(n_frames, height, width, image_folder, label_id, label_dict)
        pool = Pool(n_workers) 
        list(tqdm(pool.imap(worker, patients),total=len(patients)))
        
    finally: # To make sure processes are closed in the end, even if errors happen
        pool.close()
        pool.join()    
  
    return patient_collection



def construct_movement_labels(patient, movement_labels, use_extent=False, buffer=0):
    """ construct dataframe of movement labels over time """
    
    # create empty data frame for frame labels
    if not hasattr(patient,'frame_labels'):
        time = range( patient.video_details['n_frames'] )* (1 / patient.video_details['fps'])
        
        patient.frame_labels = pd.DataFrame(data=time, columns=['time']) 
        
        # adding column to indicate which frames have been sampled
        patient.frame_labels['pre_sampled'] = np.zeros(patient.frame_labels.shape[0])


    # loop over movement times
    for i in range(patient.movement_times.shape[0]):
        
        # adding buffer time to allow for errors in timing
        start = patient.movement_times.loc[i,'start'] + buffer
        end = patient.movement_times.loc[i,'end'] - buffer
        
        # define movement 
        movement = patient.movement_times.loc[i,'movement']
        
        # if extent of movement included then add
        if use_extent: 
            extent = patient.movement_times.loc[i,'extent']
            label = movement_labels[(movement_labels['movement']==movement) & (movement_labels['extent']==extent)]['label']
        else:
            label = movement_labels[(movement_labels['movement']==movement)]['label']
            
        # extracting label   
        label = label.values[0]
              
        # assign all rows                 
        patient.frame_labels.loc[(patient.frame_labels.time >= start) &
                                 (patient.frame_labels.time < end),'movement'] = label
        
    
    return patient



def construct_extent_labels(patient, extent_labels, buffer=0):
    """ construct dataframe of movement labels over time """
    
    # create empty data frame for frame labels
    if not hasattr(patient,'frame_labels'):
        time = range( patient.video_details['n_frames'] )* (1 / patient.video_details['fps'])
        
        patient.frame_labels = pd.DataFrame(data=time, columns=['time']) 
        
        # adding column to indicate which frames have been sampled
        patient.frame_labels['pre_sampled'] = np.zeros(patient.frame_labels.shape[0])


    # loop over movement times
    for i in range(patient.extent_times.shape[0]):
        
        # adding buffer time to allow for errors in timing
        start = patient.extent_times.loc[i,'start'] + buffer
        end = patient.extent_times.loc[i,'end'] - buffer
        
        # define movement 
        extent = patient.extent_times.loc[i,'extent']
        
        # extract label
        label = extent_labels[(extent_labels['extent']==extent)]['label']

        # extracting label   
        label = label.values[0]
              
        # assign all rows                 
        patient.frame_labels.loc[(patient.frame_labels.time >= start) &
                                 (patient.frame_labels.time < end),'extent'] = label
        

    return patient




def construct_data_dictionary(image_folder, label_dicts, label_ids):
    """ constructing data dictionary for access from deep learning model """
    
    # get paths of all images
    image_paths = list(paths.list_images(image_folder))
    
    # create empty dataframe
    data = pd.DataFrame(columns=['image_path','target','target_name'])
    
    # loop over each image and extract the folder (label) 
    for i, image_path in tqdm( enumerate(image_paths), total=len(image_paths) ):
    
        # movement name
        movement_name = os.path.normpath(image_path).split(os.path.sep)[-1]
        movement_name = '_'.join(movement_name.split('_')[1:3])
        
        # taking the directory name as the label name
        label_name = os.path.normpath(image_path).split(os.path.sep)[-2]
        
        # if two labels
        if isinstance(label_dicts, list):     
            # contruct label as empty array
            label = np.zeros(len(label_dicts)).tolist()
            
            # loop over label dictionaries
            for j, label_dict in enumerate(label_dicts):
                for k, label_ in enumerate(label_dict[label_ids[j]]):   
                    if label_ in label_name:
                        label[j] = int(k)
                        
        else:            
            # extracting the associated label integer
            if 'movement' in label_dicts:
                label = label_dicts[(label_dicts['movement']==movement_name) & (label_dicts[label_ids]==label_name)].label.values[0]
            else:
                label = label_dicts[label_dicts[label_ids]==label_name].label.values[0]
            
        # setting data to dataframe
        data.loc[i,'image_path'] = image_path
        data.loc[i,'target'] = label
        data.loc[i,'target_name'] = label_name
    
    
    data.to_csv(image_folder + 'data.csv', index=False)  

    return data

