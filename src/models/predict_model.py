import librosa
import tensorflow as tf
import numpy as np
import hydra 

# obtain audio clip sample
# process sample
# test on each individual part
# majority vote for every one second 

SAVED_MODEL_PATH = "../../models/model.keras"
SAMPLES_TO_CONSIDER = 22050

class prediction:
    model = None
    _mappins = [
        "Music",
        "Speech"
    ]
    
    _istance = None
    
    def preprocess():
        
        window_size = 0.25
        sample_rate = 22050
        hop_length = (sample_rate * window_size)/2
        n_mfcc =  13
        n_fft =  2048
        window_size_sample = 0.25 *22050

        data = np.array()
        # load audio file
        signal, _ = librosa.load(file_path, sr = sample_rate)
        
        if len(signal) < sample_rate:
            for start in range(0, len(signal), hop_length):
                end = start + window_size_sample
                if end <= len(signal):
                    segment = signal[start:end]
                    # Extract MFCCS
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,hop_length=hop_length)
                    
                    # signal = signal[:sample_rate] # ensure consistency of the length of the signal
                    
        
                    # Store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())

        
        return data

    def predict(self, file_path): 
        
        # extract MFCCs
        MFCCs = self.preprocess(file_path)
        
        # Turn into 4-dim array to feed model for predictionL (# samples, # time steps, # coefficients, label)
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        
        # predicted label
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        
        return predicted_keyword

def audio_classifier():
    
    # ensure that we only have 1 instance of KSS