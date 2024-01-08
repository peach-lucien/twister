import numpy as np
import pandas as pd

from twister.statistics.operation_class import OperationClass

featureclass_name = "AngleSymmetry"


class AngleSymmetry(OperationClass):
    
    
    def compute_features(self):
        # adding movement correlation features
                
        # perform symmetry operations on the probability of movement
        angle_predictions = pd.concat([u['predictions'] for u in self.twister_predictions['mediapipe']]).reset_index(drop=True)
        
        feats = ['anteroretrocollis', 'torticollis', 'laterocollis', 'shoulder_angle']
        angle_predictions = angle_predictions[feats] 
        
        # empty dataframe for storage of symmetries
        symmetries = pd.DataFrame(index=[0])

        # compute symmetries for head angles
        head_angles = ['anteroretrocollis', 'torticollis', 'laterocollis']

        # loop over symmetry comparisons
        for angle in head_angles:                
            
            # extract sum of times that an angle is positive or negative
            x = sum(angle_predictions[angle]<0)
            y = sum(angle_predictions[angle]>0)
            
            if x==0 and y==0:
                x=np.nan
                y=np.nan    
                
            # calculate symmetry
            symmetry = (x-y)/(x+y) # closer to 0 better
            
            # store symmetry as a feature
            #self.features['symmetry']['_'.join(movement)] = symmetry
            symmetries.loc[0,'symmetry_' + angle] = symmetry
            self.features['symmetry_' + angle] = symmetry
           
        mean_shoulder_angle = pd.Series(data=angle_predictions['shoulder_angle'].mean(), index=['mean_shoulder_angle'])
        self.features['shoulder_angle'] = mean_shoulder_angle
        # self.features['feature_vector'] = pd.concat([symmetries],axis=1)
        
        self.features['feature_vector'] = pd.DataFrame(pd.concat([symmetries.T,
                                                                  mean_shoulder_angle],axis=0)).T
        
        
        
        
        
       
        
