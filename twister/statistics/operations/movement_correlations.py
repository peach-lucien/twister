import numpy as np
import pandas as pd

from twister.statistics.operation_class import OperationClass


featureclass_name = "MovementCorrelations"



class MovementCorrelations(OperationClass):   
    
      
    def compute_features(self):
        # adding movement correlation features
        
        movement_predictions = pd.concat([u['predictions'] for u in self.twister_predictions['movement']]).reset_index(drop=True)
        
        corr = get_correlation_matrix(movement_predictions)
        
        # add correlation matrix
        self.features['correlation_matrix'] = corr
        
        # mean correlation per movement (not including diagonal)  
        mean_corr_movement = corr.mask(np.eye(corr.shape[0], dtype = bool)).mean()
        self.features['mean_correlation_per_movement'] = mean_corr_movement

        # mean correlation overall (not including diagonal) 
        mean_corr = corr.mask(np.eye(corr.shape[0], dtype = bool)).mean().mean()
        self.features['mean_correlation'] = mean_corr
        
        keep = np.triu(np.ones(corr.shape),k=1).astype('bool').reshape(corr.size)
        corr_flat = corr.stack(dropna=False)[keep]
        corr_flat.index = ['correlation_' + "_".join(a) for a in corr_flat.index.to_flat_index()]
        
        mean_corr_movement.index = ['correlation_mean_' + a for a in mean_corr_movement.index]
        mean_corr = pd.Series(data=mean_corr, index=['correlation_movement_mean'])
        
        self.features['feature_vector'] = pd.DataFrame(pd.concat([corr_flat, mean_corr_movement, mean_corr],axis=0)).T
        
        
def get_correlation_matrix(movement_predictions):    
    """ get correlation matrix """
    return movement_predictions.corr()

        
        
        