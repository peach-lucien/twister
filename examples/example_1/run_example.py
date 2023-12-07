
from twister.twister import twstr
import sys

def main():
    """ running example of twister on videos """
    
    # get paths of videos
    data_path = './data/'    
    
    # initiate twstr object with path to videos
    tw = twstr(video_path=data_path,)

    # run main twister function
    
    # preprocess videos
    tw.preprocess_videos()
    
    # construct patient objects
    tw.load_data()
    
    # make deep learning predictions
    tw.predict(save=True)
    
    # compute statistics for each patient
    tw.analyse()
    
    # aggregate feature vectors
    tw.aggregate()

    
    return

if __name__ == '__main__':
    sys.exit(main())



