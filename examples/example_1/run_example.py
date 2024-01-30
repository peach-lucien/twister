
from twister.twister import twstr
import sys

from twister.io import save_dataset, load_dataset


def main():
    """ running example of twister on videos """
    
    # get paths of videos
    data_path = './data/'    
    
    # initiate twstr object with path to videos
    #tw = twstr(video_path=data_path,)
    #tw = twstr()
    
    #tw.run()
    
    # run main twister function
    
    tw = twstr(video_path=data_path,)
    tw.preprocess_videos()

    #video_files = [data_path + 'CRF43_03m_postOP.mp4',
    #               data_path + 'HEI_31_03m_postOP.mp4',]
    #tw = twstr(video_files=video_files)

    # preprocess videos

    #tw.video_files = video_files
    
    # construct patient objects
    tw.load_data()
    
    # make deep learning predictions
    tw.predict(save=True)
    
    # compute statistics for each patient
    tw.analyse()
    
    # save dataset
    path = './'
    save_dataset(tw,'test',folder=path)   
    
    # aggregate feature vectors
    tw.aggregate()
    
    # save feature matrix to csv
    feature_matrix = tw.feature_matrix.to_csv('features.csv')
    
    return

if __name__ == '__main__':
    sys.exit(main())



