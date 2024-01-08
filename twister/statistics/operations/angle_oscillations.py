import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter

import nolds
from pycatch22 import catch22_all

from twister.statistics.operation_class import OperationClass

featureclass_name = "AngleOscillations"


class AngleOscillations(OperationClass):
    
    
    def compute_features(self):
        # adding movement correlation features
                
        # perform symmetry operations on the probability of movement
        angle_predictions = pd.concat([u['predictions'] for u in self.twister_predictions['mediapipe']]).reset_index(drop=True)
        
        feats = ['anteroretrocollis', 'torticollis', 'laterocollis']
        angle_predictions = angle_predictions[feats]
        
        if not (~angle_predictions.isna()).any().any():
            self.features['feature_vector'] = None
            return
        
        fs = self.patient.video_details[0]['fps']
        
        # empty dataframe for storage of oscillations
        oscillations = pd.DataFrame(index=[0])
        catch22 = []

        # loop over angles
        for angle in angle_predictions.columns:     
            
            signal = angle_predictions[angle]
            
            # only take non nan parts of signal
            signal = signal[~signal.isna()]
            
            signal = pd.Series(butter_bandpass_filter(signal, 2, (fs/2)-1, fs, order=6))
            
            dominant_frequency, amplitude = frequency_amplitude(signal, [2,10], fs)
            
            oscillations[angle+'_oscillation_dominant_frequency'] = np.median(dominant_frequency)
            oscillations[angle+'_oscillation_amplitude'] = np.median(amplitude)
 
            oscillations[angle+'_approximate_entropy'] = approximate_entropy(signal)   
            
            catch22.append(compute_catch22(signal, angle))
            
        catch22 = pd.concat(catch22,axis=1).reset_index(drop=True)

        self.features['feature_vector'] = pd.concat([oscillations, catch22],axis=1)
       
        
       
def approximate_entropy(timeseries, remove_nan=True):
    """ Computing approximate entropy of timeseries """    

    if remove_nan:
        timeseries = timeseries.dropna()
    
    if timeseries.shape[0]>3:
        entropy = nolds.sampen(timeseries)
    else: 
        entropy = np.nan
            
    
    return entropy
       
def compute_catch22(signal, angle):
    """ Computing catch22 features """

    catch22_computation = pd.DataFrame(catch22_all(signal)).T
    catch22_computation.columns = angle + '_' + catch22_computation.loc['names']
    catch22_computation = catch22_computation.drop('names')

    return catch22_computation



def butter_bandpass(lowcut, highcut, fs, order=5):
    return butter(order, [lowcut, highcut], fs=fs, btype='band')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y
       

def frequency_amplitude(data, f, fs):
    
    data = data.interpolate()

    x = data  
    x = np.pad(x,(int(fs/2),int(fs/2)),mode='symmetric')   
        

    min_freq = float(f[0])
    max_freq = float(f[1])

    #spectrums = []
    amplitudes = []
    frequencies = []    

    for i in range(len(data)): 
        #freqs, spectrum = compute_spectrum(x[i:i+int(fs)], fs=fs)        
        #spectrums.append(spectrum)        
        xs = x[i:i+int(fs)]
        n_samples = len(xs)
        amplitude = 2/n_samples * abs(np.fft.fft(xs))
        amplitude = amplitude[1:int(len(amplitude)/2)]     
        
        frequency  = (np.fft.fftfreq(n_samples) * n_samples * 1 / (1/fs*len(xs)))
        frequency = frequency[1:int(len(frequency)/2)]   

        freq_limits = (min_freq<=frequency) & (max_freq>=frequency)
        amplitude = amplitude[freq_limits]
        frequency = frequency[freq_limits]        
        
        amplitudes.append(np.max(amplitude))
        frequencies.append(frequency[np.argmax(amplitude)])
              
    #spectrogram = np.vstack(spectrums)
    #dominant_frequency = freqs[np.argmax(spectrogram,axis=1)]
    #amplitude = np.max(spectrogram,axis=1)
    amplitude = np.asarray(amplitudes)
    dominant_frequency = np.asarray(frequencies)

    return dominant_frequency, amplitude