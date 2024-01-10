import librosa
import tensorflow as tf
import keras
import numpy as np
import hydra 
import os
from omegaconf import DictConfig, OmegaConf

class _Audio_Classifer:
    
    model = None
    _instance = None
    
    _mappings = [
        "Music",
        "Speech"
    ]
    
    
    def __init__(self, cfg):
        self.cfg = cfg
        
    # @staticmethod
    # def load_data(file_path):
    #     """_summary_
    #     Loads file 
    #     Args:
    #         cfg (DictConfig): Configuration object containing:
    #             data_path (str): Path to json file containing data 
    #     Returns:
    #         :return X (ndarray): Inputs
    #         :return y (ndarray): Targets
    #     """
    #     file = os.path.basename(file_path)
    #     try:
    #         with open(file_path, "r") as fp:
    #             data = json.load(fp)
    #     except Exception as e:
    #         print(f'Error loading data: {e}')
    #         return
        
    #     return file
    
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
        prediction_array = np.array(mfcc_list, dtype = np.float32)
        
        return prediction_array
                    
    def predict(self, file_path):
        
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
            probabilities = self.sigmoid(logits)
            print(probabilities)
            predicted_class = self._mappings[int(probabilities > 0.5)]
            all_predictions.append(predicted_class)
            print(f'Segment {count} Prediction: {predicted_class}')
            
        return all_predictions
                    
# @hydra.main(version_base=None, config_path='config', config_name='config') 
# def Audio_Classifier(cfg: DictConfig):    
#     """Factory function for Audio Classifier Class 

#     :return _Audio_Classifier._instance:
#     """
  
#     if _Audio_Classifer._instance is None:
#         _Audio_Classifer._instance = _Audio_Classifer(cfg)
#         _Audio_Classifer.model = tf.keras.models.load_model(cfg.predictions.saved_model_path)
#     return _Audio_Classifer._instance


# file_path =  "data/raw/prediction_data/music_predict"

# if __name__ == "__main__":
#     # creating 2 instances of audio classifier
    
#     ac = Audio_Classifier()
#     ac1 = Audio_Classifier()
#     print(ac)
#     assert ac is ac1

#     # make prediction
#     prediction = ac.predict("data/raw/prediction_data/music_predict")
#   


# cfg = OmegaConf.load("src/config/config.yaml")
# print(cfg.predictions.data_path.music)

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg:DictConfig):
    # file = "data/raw/prediction_data/music_predict/predict.wav"
    file = cfg.predictions.data_path.music
    # file = cfg.predictions.data_path.speech
    classifier = _Audio_Classifer(cfg)
    classifier.model = tf.keras.models.load_model(cfg.predictions.saved_model_path)
    # print(classifier.model)
    prediction = classifier.predict(file) 
    
    print(prediction)
    print(file)
    
if __name__ == "__main__":
    main()