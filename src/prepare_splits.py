import librosa
import os
import json
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


def collect_clips(dataset_path):
    """_summary_

    Args:
        dataset_path (_type_): path to data set

    Returns:
        data set formatted
    """
    clips = []
    for dirpath, _, filenames in os.walk(dataset_path):
        if dirpath != dataset_path:  # Ignore the root directory
            label = dirpath.split("/")[-1]
            for file in filenames:
                file_path = os.path.join(dirpath, file)
                clips.append((file_path, label))
    return clips



def split_clips(clips, test_size=0.2, validation_size=0.2):
    """_summary_

    Args:
        clips (_type_): formatted unprocessed dataset 
        test_size (float, optional): test size for split, defaults to 0.2.
        validation_size (float, optional): validation size for splits, defaults to 0.2.

    Returns:
        train, validation, test clips
    """
    # Split into training and test sets
    train_clips, test_clips = train_test_split(clips, test_size=test_size)
    # Split training into training and validation sets
    train_clips, validation_clips = train_test_split(train_clips, test_size=validation_size)
    return train_clips, validation_clips, test_clips


@hydra.main(version_base=None, config_path='config', config_name='config')
def preprocess_dataset(cfg: DictConfig):
    """_summary_
    Preprocesses dataset into json format with librosa by obtaining MFCCs for each clip. 
    Creates, train, test, and validation data for each window size specified in 
    config file. 
    Args:
        cfg (DictConfig): _description_
    """
    dataset_path = cfg.dataset.path
    json_path = cfg.dataset.json_path
    samples_to_consider = cfg.dataset.samples_to_consider
    n_mfcc = cfg.dataset.n_mfcc
    n_fft = cfg.dataset.n_fft
    windows = cfg.dataset.windows
    sample_rate = cfg.dataset.sample_rate
    window_sizes = {window['label']: int(window['duration'] * sample_rate) for window in windows}
    hop_lengths = {label: int(size / 2) for label, size in window_sizes.items()}
    label_mapping = {
                                "music_wav": 0,
                                "speech_wav": 1
                            }
    
    # Collect and split clips
    all_clips = collect_clips(dataset_path)
    train_clips, validation_clips, test_clips = split_clips(all_clips)
    for window_label, window_size in window_sizes.items():
    # Process each set
        for set_name, clips in [("train", train_clips), ("validation", validation_clips), ("test", test_clips)]:
            data = {
                "mappings": [],
                "labels": [],
                "MFCCs": [],
                "files": []
            }

            for file_path, label in clips:
                temp = []
            
                
                signal, _ = librosa.load(file_path, sr = sample_rate) 
                
                # dropping audio files with less than pre-decided number of samples 
                if len(signal) >= sample_rate: 
                    hop_length = hop_lengths[window_label]
                    window_size_samples = int(window_size * sample_rate)
                    
                    # print("\nProcessing: '{}'".format(label))
                    for start in range(0, len(signal), hop_length):
                        end = start + window_size_samples 
                        if end <= len(signal):
                            segment = signal[start:end]
                            # Extract MFCCS
                            MFCCs = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft,hop_length=hop_length)
                        
                            # signal = signal[:sample_rate] # ensure consistency of the length of the signal
                            
                
                            # Store data for analysed track
                            data["mappings"].append(label)
                            data["MFCCs"].append(MFCCs.T.tolist())
                            data["labels"].append(label_mapping[label])
                            data["files"].append(file_path)
                            # print("{}: {}".format(file_path, label_mapping[label]))
            # print(data["MFCCs"]) # To check outputs 
    
    # save data in json file

    # Save data in a JSON file specific to the set
            mfccs_array = np.array(data["MFCCs"])
            print(f'{window_label} {set_name}s Shape is: {mfccs_array.shape}')
            json_path = cfg.dataset.json_path.replace(".json", f"_{set_name}_{window_label}.json")
            with open(json_path, "w") as fp:
                json.dump(data, fp, indent=4)
            return
        
if __name__ == "__main__": 
    preprocess_dataset()