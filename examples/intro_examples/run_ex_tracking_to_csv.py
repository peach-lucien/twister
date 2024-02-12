
from twister.twister import twstr
import sys

from twister.io import save_dataset, load_dataset, load_from_csv


def main():
    """ running example of twister on videos """
    
    # get paths of videos
    data_path = './data/'    
    
    # initiate twstr object with path to videos
    tw = twstr(video_path=data_path,)
    
    # run main twister function
    tw.run()    

    # access the feature matrix and save to csv
    tw.feature_matrix.to_csv('features.csv')
    
    return

if __name__ == '__main__':
    sys.exit(main())



