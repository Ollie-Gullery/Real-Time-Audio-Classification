import librosa
import os
import json
import numpy as np

DATASET_PATH = "data/raw/dataset"
JSON_PATH = "data/processed/data.json"
SAMPLES_TO_CONSIDER = 22050 * 30 # 1 sec worth of loading audio with librosa is 22050 samples/sec, multiply by 30 for 30 seconds, standardizes length of the audio files


def preprocess_dataset(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """_summary_
    Extracts MFCCs from music dataset and saves them into a json file located in our processed data folder 
  
    Args:
        dataset_path (_type_): _description_
        json_path (_type_): _description_
        n_mfcc (int, optional): _description_. Defaults to 13.
        n_fft (int, optional): _description_. Defaults to 2048.
        hop_length (int, optional): _description_. Defaults to 512.
    """
    # data dictoinary
    
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        

        # ensure we're at sub-folder level
        if dirpath != dataset_path:
            
            # save label (i.e., sub-folder name) in the mapping
            label = dirpath.split("/")[-1]
            data["mappings"].append(label)
            print("\nProcessing: '{}'".format(label))
            
            # process all audio files in sub-dir and store MFCCs
            for file in filenames:
                temp = []
                
                file_path = os.path.join(dirpath, file)
                
                signal, sample_rate = librosa.load(file_path) 
                
                # dropping audio files with less than pre-decided number of samples 
                if len(signal) >= SAMPLES_TO_CONSIDER: 
                    
                    signal = signal[:SAMPLES_TO_CONSIDER] # ensure consistency of the length of the signal
                    
                    # Extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,hop_length=hop_length)
                    
                    # Store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
          
            # print(data["MFCCs"]) # To check outputs 
    
    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

# Ensures method is only executed when script is run directly, not when it is imported as a module in another script
if __name__ == "__main__": 
    preprocess_dataset(DATASET_PATH, JSON_PATH)