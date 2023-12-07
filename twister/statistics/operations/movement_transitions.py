

import numpy as np
import pandas as pd

from twister.statistics.operation_class import OperationClass


featureclass_name = "MovementTransitions"

class MovementTransitions(OperationClass):
    
    
    def compute_features(self):
        # adding movement correlation features
        
        # extracting probabilities of movements
        #movement_predictions = self.twister_predictions['movement']['probabilities']
        movement_predictions = pd.concat([u['probabilities'] for u in self.twister_predictions['movement']]).reset_index(drop=True)
        
        # defining time window for averaging
        window = 1
        
        # rolling average window to smooth out labels
        probabilities = movement_predictions.rolling(window).mean()         
        probabilities = probabilities.iloc[::window,:]
        probabilities = probabilities.dropna(axis=0)
        
        # predicted class label names
        predicted_class = pd.DataFrame(probabilities.idxmax(axis=1))
        
        # construct transition matrix
        T, n_transitions = calculate_transition_matrix(predicted_class.copy())
     
        # add column names
        T = T +  pd.DataFrame(data=np.zeros([7,7]), index=movement_predictions.columns, columns=movement_predictions.columns)
        T = T.fillna(0)
        
        n_transitions = n_transitions +  pd.DataFrame(data=np.zeros([7,7]), index=movement_predictions.columns, columns=movement_predictions.columns)
        n_transitions = n_transitions.fillna(0)
        
        # if transition never observed to a movement - then set 1 on diagonal
        for i, col in enumerate(T.columns):
            if (T.loc[:,col]==0).all():
                T.loc[col,col]=1           
                
        # return transition matrix
        self.features['transition_matrix'] = T    
        self.features['n_transitions'] = n_transitions       
        
        # flatten into features vector
        #keep = np.triu(np.ones(T.shape),k=1).astype('bool').reshape(T.size)
        T_flat = T.stack(dropna=False)#[keep]
        T_flat.index = ['transition_probability_' + "_".join(a) for a in T_flat.index.to_flat_index()]
        
        self.features['feature_vector'] = pd.DataFrame(data=T_flat).T       
    

def calculate_transition_matrix(df):
    """cConstructing transition matrix """
    
    # shift function to shift movements by one step
    df['shift'] = df[0].shift(-1)
    
    # add a count column (for group by function)
    df['count'] = 1    
    
    # groupby and then unstack, fill the zeros
    sum_mat = df.groupby([0, 'shift']).count().unstack().fillna(0) 
    
    # normalise by occurences and save values to get transition matrix
    trans_mat = sum_mat.div(sum_mat.sum(axis=1), axis=0)   
    
    trans_mat.columns = trans_mat.columns.droplevel(0)
    sum_mat.columns = sum_mat.columns.droplevel(0)
    
    return trans_mat, sum_mat

