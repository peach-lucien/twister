
from twister.statistics.operation_class import OperationClass

import pandas as pd

featureclass_name = "Angles"

class Angles(OperationClass):   
    
      
    def compute_features(self):        
        
        angle_predictions = pd.concat([u['predictions'] for u in self.twister_predictions['mediapipe']]).reset_index(drop=True)

        angle_predictions = angle_predictions[['anteroretrocollis',
                                            'torticollis',         
                                            'laterocollis' ,         
                                            'shoulder_angle',]]

        mean_angles = angle_predictions.mean()
        mean_angles.index = ['mean_head_angle_' + u for u in mean_angles.index ]
        self.features['mean_angles'] = mean_angles 

        median_angles = angle_predictions.median()
        median_angles.index = ['median_head_angle_' + u for u in median_angles.index ]
        self.features['median_angles'] = median_angles 
        
        std_angles = angle_predictions.std()
        std_angles.index = ['std_head_angle_' + u for u in std_angles.index ]
        self.features['std_angles'] = std_angles 

        skewness_angles = angle_predictions.skew()
        skewness_angles.index = ['skewness_head_angle_' + u for u in skewness_angles.index ]
        self.features['skewness_angles'] = skewness_angles 

        kurtosis_angles = angle_predictions.kurtosis()
        kurtosis_angles.index = ['kurtosis_head_angle_' + u for u in kurtosis_angles.index ]
        self.features['kurtosis_angles'] = kurtosis_angles 
                
        self.features['feature_vector'] = pd.DataFrame(pd.concat([mean_angles, median_angles, 
                                                                  std_angles, skewness_angles,
                                                                  kurtosis_angles],axis=0)).T

        

        
        
        

        
        

        
        
        