import librosa
import os
import json
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path='config', config_name='config')
def preprocess_dataset(cfg: DictConfig):
    """
    _summary_
    Extracts MFCCs from music dataset and saves them into a json file located in our processed data folder 
  
    Args:
        cfg (DictConfig): Configuration object containing:
            dataset_path (_type_): _description_
            json_path (_type_): _description_
            n_mfcc (int, optional): _description_. Defaults to 13.
            n_fft (int, optional): _description_. Defaults to 2048.
            hop_length (int, optional): _description_. Defaults to 512.
    """
    dataset_path = cfg.dataset.path
    json_path = cfg.dataset.json_path
    samples_to_consider = cfg.dataset.samples_to_consider
    n_mfcc = cfg.dataset.n_mfcc
    n_fft = cfg.dataset.n_fft
    hop_length = cfg.dataset.hop_length
    
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        

        # ensure we're at sub-folder level
        if dirpath is not dataset_path:
            
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
                if len(signal) >= samples_to_consider: 
                    
                    signal = signal[:samples_to_consider] # ensure consistency of the length of the signal
                    
                    # Extract MFCCs
                    MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,hop_length=hop_length)
                    
                    # Store data for analysed track
                    data["MFCCs"].append(MFCCs.T.tolist())
                    data["labels"].append(i-1)
                    data["files"].append(file_path)
                    print("{}: {}".format(file_path, i-1))
            # print(data["MFCCs"]) # To check outputs 
    
    # save data in json file
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)

# Ensures method is only executed when script is run directly, not when it is imported as a module in another script
if __name__ == "__main__": 
    preprocess_dataset()
    
    