import librosa
import tensorflow as tf
import keras
import numpy as np
import hydra 
import os
from omegaconf import DictConfig, OmegaConf
import matplotlib.pyplot as plt
class _Audio_Classifier:
    

    _instance = None # singleton design pattern 
    
    _mappings = [
        "Music",
        "Speech"
    ]
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(_Audio_Classifier, cls).__new__(cls)
            cls._instance.initialized = False
            
        return cls._instance
    
    def __init__(self, cfg):
        if not self.initialized:  
            self.cfg = cfg
            self.model = tf.keras.models.load_model(cfg.predictions.saved_model_path)
            self.initialized = True

    
    @staticmethod
    def sigmoid(x):
        return 1/ (1+np.exp(-x))
    
    def preprocess(self, file_path):
        """_summary_
        Preprocesses singular file with librosa by obtaining MFCCs for specified file
        
        Args:
            cfg (DictConfig): _description_ in config.yaml
        """
        
        # obtain variables
        n_mfcc = self.cfg.predictions.n_mfcc
        n_fft = self.cfg.predictions.n_fft
        window_size = self.cfg.predictions.window_size
        sample_rate = self.cfg.predictions.sample_rate
        hop_length = self.cfg.predictions.hop_length
        
        # specify window sample size
        window_sample_size = int(window_size * sample_rate)
        
        # Load file with librosa 
        signal, _ = librosa.load(file_path, sr = sample_rate)
        
        mfcc_list = []
        
        if len(signal) >= window_sample_size:
            for start in range(0, len(signal), window_sample_size):
                end = start + window_sample_size
                if end <= len(signal):
                    segment = signal[start:end]
                    MFCCs  = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
                    mfcc_list.append(MFCCs.T)
        
        # Creating Prediction Array
        MFCCs_array = np.array(mfcc_list, dtype = np.float32)
        
        return MFCCs_array
    
    def predict_signal(self, signal, is_MFCC = False):
        """_summary_
        Predicts between music or speech for a singular signal
        Args:
            signal (_type_): _description_
            is_MFCC (bool, optional): _description_. Defaults to False.

        Returns:
            _type_: predicted class 
        """
        # obtain variables
        n_mfcc = self.cfg.predictions.n_mfcc
        n_fft = self.cfg.predictions.n_fft
        window_size = self.cfg.predictions.window_size
        sample_rate = self.cfg.predictions.sample_rate
        hop_length = self.cfg.predictions.hop_length
        if is_MFCC == False:
            mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
             
        else:
            mfcc = signal
        mfcc = mfcc[np.newaxis, ..., np.newaxis]
        mfcc = mfcc.reshape(-1, 11, 13, 1) # reshaping as pyaudio signal different to librosa which was used to train model
        logits = self.model.predict(mfcc)
        probability = self.sigmoid(logits)
        predicted_class = self._mappings[int(probability > self.cfg.predictions.threshold)]
        print(f'Prediction {predicted_class} | Prediction Probability {probability}')
        return predicted_class
                    
    def predict(self, file_path):
        """_summary_
        Classifies audio file based on 0.25s second segments 
        Args:
            file_path (_type_): Path for audio file

        Returns:
            _type_: Predictions
        """
        MFCCs = self.preprocess(file_path)
        all_predictions = []
        count = 0
        # We need 4 dimensional array for predictions: (# samples, # time steps, # coefficients, 1)
        for mfcc in MFCCs:   
            count += 1
            # We need 4 dimensional array for predictions: (# samples, # time steps, # coefficients, 1)
            mfcc = mfcc[np.newaxis, ..., np.newaxis] 
        
            # get predicted label
            logits = self.model.predict(mfcc)
            probability = self.sigmoid(logits)
            print(probability)
            predicted_class = self._mappings[int(probability > self.cfg.predictions.threshold)]
            all_predictions.append(predicted_class)
            print(f'Segment {count} | Prediction: {predicted_class} | Probability {probability}')
            
        return all_predictions


    def plot_mfcc(self, mfcc):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfcc[i], x_axis='time', sr=self.cfg.predictions.sample_rate)
        plt.colorbar()
        plt.title('MFCC')
        plt.tight_layout()
        plt.show()
def Audio_Classifier():    
    """Factory function for Audio Classifier Class 

    :return _Audio_Classifier._instance:
    """
    
    cfg = OmegaConf.load("src/config/config.yaml")
    
    return _Audio_Classifier(cfg)

# file_path =  "data/raw/prediction_data/speech_predict/speech_predict.wav"
file_path='output.wav'

if __name__ == "__main__":
    # creating 2 instances of audio classifier
    
    ac = Audio_Classifier()
    ac1 = Audio_Classifier()
    
    assert ac is ac1 # Ensures that ac and ac1 are the same instance (singleton pattern is working)
    # prediction = ac.predict(file_path)
    # make prediction
    print("Making Prediction")

    
    # single_prediction = ac.predict_signal(mfcc, is_MFCC=True)
    mfcc = ac.preprocess(file_path)
    print("looping predictions")
    for i in range(11):
        ac.plot_mfcc(mfcc)
        print(mfcc[i].shape)
        single_prediction = ac.predict_signal(mfcc[i], is_MFCC=True)
        print(f"Single Prediction: {single_prediction}")

    
  


