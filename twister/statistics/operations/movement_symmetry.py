import numpy as np
import pandas as pd

from twister.statistics.operation_class import OperationClass

featureclass_name = "MovementSymmetry"


class MovementSymmetry(OperationClass):
    
    
    def compute_features(self):
        # adding movement correlation features
        
        
        # movement symmetries to compare
        movement_symmetries = [['head_back', 'head_chest'], ['rot_left', 'rot_right'],['tilt_left', 'tilt_right']]

        # perform symmetry operations on the probability of movement
        #movement_predictions = self.twister_predictions['movement']['probabilities'].idxmax(axis=1)
        movement_predictions = pd.concat([u['probabilities'] for u in self.twister_predictions['movement']]).reset_index(drop=True).idxmax(axis=1)
    
        # empty dataframe for storage of symmetries
        symmetries = pd.DataFrame(columns=['symmetry_' + '_'.join(u) for u in movement_symmetries],index=[0])

        # loop over symmetry comparisons
        for movement in movement_symmetries:                
            
            # extract sum of times we observe a particular movement
            x = sum(movement_predictions==movement[0])
            y = sum(movement_predictions==movement[1])
            
            if x==0 and y==0:
                x=np.nan
                y=np.nan    
                
            # calculate symmetry
            symmetry = (x-y)/(x+y) # closer to 0 better
            
            # store symmetry as a feature
            #self.features['symmetry']['_'.join(movement)] = symmetry
            symmetries.loc[0,'symmetry_' + '_'.join(movement)] = symmetry
            self.features['symmetry_' + '_'.join(movement)] = symmetry
           
        # get movements
        movements = self.twister_predictions['movement'][0]['probabilities'].columns  
        
        # look at time spent in each movement
        movement_n = {}
        for movement in movements:
            
            # total time in certain movement observed
            movement_n[movement] = sum(movement_predictions==movement) / self.patient.sampling_frequency           
    
        # store sum of movements
        movement_n = pd.DataFrame(movement_n,index=[0])
        self.features['sum_movements'] = movement_n
         
        # flatten into features vector
        movement_n.columns = ['movement_n_' + a for a in movement_n.columns]
        self.features['feature_vector'] = pd.concat([movement_n, symmetries],axis=1)
       
        
