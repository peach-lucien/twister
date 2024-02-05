
from twister.statistics.operation_class import OperationClass

import pandas as pd
import numpy as np 

featureclass_name = "FacialExpressions"


class FacialExpressions(OperationClass):   
    
      
    def compute_features(self):        
        
        facial_predictions = pd.concat([u['predictions'] for u in self.twister_predictions['mediapipe']]).reset_index(drop=True)
               
        # neutral expression
        feats = ['_neutral']   
        mean_neutral_expression = pd.Series(data=facial_predictions[feats].mean()[0], index=['mean_neutral_expression']) 
        median_neutral_expression = pd.Series(data=facial_predictions[feats].median()[0], index=['median_neutral_expression']) 
        std_neutral_expression = pd.Series(data=facial_predictions[feats].std()[0], index=['std_neutral_expression']) 
        self.features['neutral_expression'] = mean_neutral_expression        
        
        
        # brow related features
        feats = ['browDownLeft', 'browDownRight', 'browInnerUp',
                 'browOuterUpLeft', 'browOuterUpRight']
        mean_brow_expression = pd.Series(data=facial_predictions[feats].std().mean(), index=['mean_brow_expression']) 
        median_brow_expression = pd.Series(data=facial_predictions[feats].median().mean(), index=['median_brow_expression']) 
        std_brow_expression = pd.Series(data=facial_predictions[feats].std().mean(), index=['std_brow_expression']) 
        self.features['brow_expression'] = mean_brow_expression 

        # eye related features
        feats =  ['eyeBlinkLeft', 'eyeBlinkRight', 'eyeLookDownLeft',
                'eyeLookDownRight', 'eyeLookInLeft', 'eyeLookInRight', 'eyeLookOutLeft',
                'eyeLookOutRight', 'eyeLookUpLeft', 'eyeLookUpRight', 'eyeSquintLeft',
                'eyeSquintRight', 'eyeWideLeft', 'eyeWideRight']
        mean_eye_expression = pd.Series(data=facial_predictions[feats].mean(axis=1).mean(), index=['mean_eye_expression'])  # mean eye deviation (mean across time)
        median_eye_expression = pd.Series(data=facial_predictions[feats].median(axis=1).mean(), index=['median_eye_expression'])  # mean eye deviation (mean across time)
        std_eye_expression = pd.Series(data=facial_predictions[feats].std(axis=1).mean(), index=['std_eye_expression'])  # mean eye deviation (mean across time)
        self.features['eye_expression'] = mean_eye_expression

        # jaw related features
        feats = ['jawForward','jawLeft', 'jawOpen', 'jawRight']
        mean_jaw_expression= pd.Series(data=facial_predictions[feats].mean(axis=1).mean(), index=['mean_jaw_expression']) # mean jaw deviation (mean across time)
        median_jaw_expression= pd.Series(data=facial_predictions[feats].median(axis=1).mean(), index=['median_jaw_expression']) # mean jaw deviation (mean across time)
        std_jaw_expression= pd.Series(data=facial_predictions[feats].std(axis=1).mean(), index=['std_jaw_expression']) # mean jaw deviation (mean across time)
        self.features['jaw_expression'] = mean_jaw_expression

        # mouth related features
        feats = [ 'mouthClose', 'mouthDimpleLeft',
                'mouthDimpleRight', 'mouthFrownLeft', 'mouthFrownRight', 'mouthFunnel',
                'mouthLeft', 'mouthLowerDownLeft', 'mouthLowerDownRight',
                'mouthPressLeft', 'mouthPressRight', 'mouthPucker', 'mouthRight',
                'mouthRollLower', 'mouthRollUpper', 'mouthShrugLower',
                'mouthShrugUpper', 'mouthSmileLeft', 'mouthSmileRight',
                'mouthStretchLeft', 'mouthStretchRight', 'mouthUpperUpLeft',
                'mouthUpperUpRight',]
        mean_mouth_expression = pd.Series(data=facial_predictions[feats].mean(axis=1).mean(), index=['mean_mouth_expression'])
        median_mouth_expression = pd.Series(data=facial_predictions[feats].median(axis=1).mean(), index=['median_mouth_expression'])
        std_mouth_expression = pd.Series(data=facial_predictions[feats].std(axis=1).mean(), index=['std_mouth_expression'])
        self.features['mouth_expression'] = mean_mouth_expression

        # nose related features
        feats = ['noseSneerLeft', 'noseSneerRight']
        mean_nose_expression = pd.Series(data=facial_predictions[feats].mean(axis=1).mean(), index=['mean_nose_expression'])
        median_nose_expression = pd.Series(data=facial_predictions[feats].median(axis=1).mean(), index=['median_nose_expression'])
        std_nose_expression = pd.Series(data=facial_predictions[feats].std(axis=1).mean(), index=['std_nose_expression'])
        self.features['nose_expression'] = mean_nose_expression

        # right related features
        feats = ['browDownRight','browOuterUpRight', 
                'cheekSquintRight', 'eyeBlinkRight', 
                'eyeLookDownRight', 'eyeLookInRight', 
                'eyeLookOutRight',  'eyeLookUpRight',
                'eyeSquintRight',  'eyeWideRight', 
                'jawRight','mouthDimpleRight', 'mouthFrownRight', 
                'mouthLowerDownRight','mouthPressRight', 'mouthRight',
                'mouthSmileRight','mouthStretchRight', 
                'mouthUpperUpRight', 'noseSneerRight']
        mean_rightward_expression = pd.Series(data=facial_predictions[feats].mean(axis=1).mean(), index=['mean_rightward_expression'])
        median_rightward_expression = pd.Series(data=facial_predictions[feats].median(axis=1).mean(), index=['median_rightward_expression'])
        std_rightward_expression = pd.Series(data=facial_predictions[feats].std(axis=1).mean(), index=['std_rightward_expression'])
        self.features['rightward_expression'] = mean_rightward_expression

        # left related features
        feats = ['browDownLeft','browOuterUpLeft', 
                'cheekSquintLeft', 'eyeBlinkLeft', 
                'eyeLookDownLeft', 'eyeLookInLeft', 
                'eyeLookOutLeft',  'eyeLookUpLeft',
                'eyeSquintLeft',  'eyeWideLeft', 
                'jawLeft','mouthDimpleLeft', 'mouthFrownLeft', 
                'mouthLowerDownLeft','mouthPressLeft', 'mouthLeft',
                'mouthSmileLeft','mouthStretchLeft', 
                'mouthUpperUpLeft', 'noseSneerLeft']
        mean_leftward_expression = pd.Series(data=facial_predictions[feats].mean(axis=1).mean(), index=['mean_leftward_expression'])
        median_leftward_expression = pd.Series(data=facial_predictions[feats].median(axis=1).mean(), index=['median_leftward_expression'])
        std_leftward_expression = pd.Series(data=facial_predictions[feats].std(axis=1).mean(), index=['std_leftward_expression'])
        self.features['leftward_expression'] = mean_leftward_expression


        # TODO: add more nuanced features based on expressions
        # oscillations of certain ticks ? 
        
        self.features['feature_vector'] = pd.DataFrame(pd.concat([mean_neutral_expression,median_neutral_expression,std_neutral_expression,
                                                                  mean_brow_expression, median_brow_expression, std_brow_expression,
                                                                  mean_eye_expression, median_eye_expression, std_eye_expression,
                                                                  mean_jaw_expression,median_jaw_expression, std_jaw_expression,
                                                                  mean_mouth_expression,median_mouth_expression, std_mouth_expression,
                                                                  mean_nose_expression,median_nose_expression, std_nose_expression,
                                                                  mean_rightward_expression,median_rightward_expression, std_rightward_expression,
                                                                  mean_leftward_expression,median_leftward_expression, std_leftward_expression
                                                                  ],axis=0)).T

        
        
        

        
        

        
        
        