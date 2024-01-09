import librosa
import os
import json
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split


class DataProcessor: 
    def __init__(self, cfg:DictConfig):
        self.cfg = cfg

    @staticmethod
    def collect_files(dataset_path):
        """_summary_

        Args:
            dataset_path (_type_): path to data set

        Returns:
            data set formatted
        """
        files = []
        for dirpath, _, filenames in os.walk(dataset_path):
            if dirpath != dataset_path:
                label = dirpath.split("/")[-1]
                for file in filenames:
                    file_path = os.path.join(dirpath, file)
                    files.append((file_path, label))
        return files
        
    def split_dataset(self, clips):
        """_summary_
        Splits the dataset into training, validation, and test sets.

        Args:
            files (list): A list of tuples with file paths and their corresponding labels.
            test_size (float, optional): The proportion of the dataset to include in the test split.
            validation_size (float, optional): The proportion of the training dataset to include in the validation split.

        Returns:
            tuple: Three lists containing the training, validation, and test sets, respectively.
        """
        test_size = self.cfg.preprocess.test_size
        validation_size = self.cfg.preprocess.validation_size
           
        # Split into training and test sets
        train_clips, test_clips = train_test_split(clips, test_size=test_size)
        # Split training into training and validation sets
        train_clips, validation_clips = train_test_split(train_clips, test_size=validation_size)
        return train_clips, validation_clips, test_clips
    
    def preprocess_dataset(self):
        dataset_path = self.cfg.preprocess.path
        n_mfcc = self.cfg.preprocess.n_mfcc
        n_fft = self.cfg.preprocess.n_fft
        window_size = self.cfg.preprocess.window_size
        sample_rate = self.cfg.preprocess.sample_rate
        hop_length = self.cfg.preprocess.hop_length
        
        window_sample_size = int(window_size * sample_rate)

        label_mapping = {
            "music_wav": 0,
            "speech_wav": 1
        }
        
        # Collect Files and Split Data
        files = self.collect_files(dataset_path)
        
        train_files, validation_files, test_files = self.split_dataset(files)
        
        # Process Data
        for set_name, files in [("train", train_files), ("validation", validation_files), ("test", test_files)]:
            data = {
                "mappings": [],
                "labels": [],
                "MFCCs": [],
                "files": []
            }
            print("\nProcessing: '{}' data".format(set_name))
            for file_path, label in files:
            
                # load files
                signal, _ = librosa.load(file_path, sr = sample_rate)
                
                # ignore audio files with less than window sample size
                if len(signal >= window_sample_size):
                     # Processing Segements of size 'window_size' 
                    for start in range(0, len(signal), window_sample_size):
                        end = start + window_sample_size
                        if end <= len(signal):
                            segment = signal[start:end]
                            # Extract MFCCs
                            MFCCs  = librosa.feature.mfcc(y=segment, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

                            
                            # Store MFCC Data for Segment
                            data["mappings"].append(label)
                            data["MFCCs"].append(MFCCs.T.tolist())
                            data["labels"].append(label_mapping[label])
                            data["files"].append(file_path)
                            print("{}: {}".format(file_path, label_mapping[label]))
            # Save Data into JSON File
            json_path = self.cfg.preprocess.json_path.replace(".json", f'_{set_name}.json')
            with open(json_path, "w") as fp:
                json.dump(data, fp, indent=4)
        
        print("Data processed successfully.")
        
        return
            
@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig):
    processor = DataProcessor(cfg)
    processor.preprocess_dataset()

if __name__ == "__main__":
    main()