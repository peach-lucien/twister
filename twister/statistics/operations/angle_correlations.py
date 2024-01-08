
from twister.statistics.operation_class import OperationClass

import pandas as pd
import numpy as np 

featureclass_name = "AngleCorrelations"



class AngleCorrelations(OperationClass):   
    
      
    def compute_features(self):        
        
        angle_predictions = pd.concat([u['predictions'] for u in self.twister_predictions['mediapipe']]).reset_index(drop=True)
        
        feats = ['anteroretrocollis', 'torticollis', 'laterocollis']
        angle_predictions = angle_predictions[feats]
        
        corr = angle_predictions.corr()
        self.features['angle_correlations'] = corr        
        keep = np.triu(np.ones(corr.shape),k=1).astype('bool').reshape(corr.size)
        corr_flat = corr.stack(dropna=False)[keep]
                
        mean_corr_angle = corr.mask(np.eye(corr.shape[0], dtype = bool)).mean()
        self.features['mean_correlation_per_angle'] = mean_corr_angle

        # mean correlation overall (not including diagonal) 
        mean_corr = corr.mask(np.eye(corr.shape[0], dtype = bool)).mean().mean()
        self.features['mean_angle_correlation'] = mean_corr  
        

        keep = np.triu(np.ones(corr.shape),k=1).astype('bool').reshape(corr.size)
        corr_flat = corr.stack(dropna=False)[keep]
        corr_flat.index = ['correlation_' + "_".join(a) for a in corr_flat.index.to_flat_index()]
        
        mean_corr_angle.index = ['correlation_angle_mean_' + a for a in mean_corr_angle.index]
        mean_corr = pd.Series(data=mean_corr, index=['correlation_angle_mean'])

        self.features['feature_vector'] = pd.DataFrame(pd.concat([corr_flat, mean_corr_angle, mean_corr],axis=0)).T

        
        
        

        
        

        
        
        