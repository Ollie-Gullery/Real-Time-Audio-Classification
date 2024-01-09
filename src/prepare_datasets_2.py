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
    windows = cfg.dataset.windows
    sample_rate = cfg.dataset.sample_rate
    window_sizes = {window['label']: int(window['duration'] * sample_rate) for window in windows}

    # Calculate hop lengths for each window size (assuming 50% overlap)
    hop_lengths = {label: int(size / 2) for label, size in window_sizes.items()}
    
    for window_label, window_size in window_sizes.items():
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
                    
                    signal, _ = librosa.load(file_path, sr = sample_rate) 
                    
                    # dropping audio files with less than pre-decided number of samples 
                    if len(signal) >= sample_rate: 
                        
                        hop_length = hop_lengths[window_label]
                        data["mappings"].append(label)
                        
                        for start in range(0, len(signal), hop_length):
                            end = start + window_size
                            if end <= len(signal):
                                segment = signal[start:end]
                                # Extract MFCCS
                                MFCCs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,hop_length=hop_length)
                            
                                # signal = signal[:sample_rate] # ensure consistency of the length of the signal
                        
                       
                                # Store data for analysed track
                                data["MFCCs"].append(MFCCs.T.tolist())
                                data["labels"].append(i-1)
                                data["files"].append(file_path)
                                print("{}: {}".format(file_path, i-1))
                # print(data["MFCCs"]) # To check outputs 
        
        # save data in json file
        json_path_modified = json_path.replace(".json", f"_{window_label}.json")
        with open(json_path_modified , "w") as fp:
            json.dump(data, fp, indent=4)

# Ensures method is only executed when script is run directly, not when it is imported as a module in another script
if __name__ == "__main__": 
    preprocess_dataset()
    
    