import librosa
import os
import json

DATASET_PATH = "dataset"
JSON_PATH = "data.json"
SAMPLES_TO_CONSIDER = 22050 # 1 sec worth of loading sound with librosa


def prepare_dataset(dataset_path, json_path, n_mfcc=13, n_fft=2048, hop_length=512):
    """_summary_
    Extracts MFCCs from music dataset and saves them into a json file
  
    Args:
        dataset_path (_type_): _description_
        json_path (_type_): _description_
        n_mfcc (int, optional): _description_. Defaults to 13.
        n_fft (int, optional): _description_. Defaults to 2048.
        hop_length (int, optional): _description_. Defaults to 512.
    """

    # data dictoinary
    
    data = {
        "mappings": ["Music", "Speech", ...],
        "labels": [0,0,1,1,...],
        "MFCCs": [],
        "files": []
    }

    # loop through all sub-dirs
    for i in os.walk(dataset_path):
        print(i)

    # ensure we're at sub-folder level

    # save label (i.e., sub-folder name) in the mapping

    # process all audio files in sub-dir and store MFCCs

    # load audio file and slice it to ensure length consistency among different files

    # drop audio files with less than pre-decided number of samples

    # ensure consistency of the length of the signal

    # extract MFCCs

    # store data for analysed track

    # save data in json file
# for i,k in os.walk("/dataset"):
#         print("test")
#         print(f'i here: {i}')
#         print(f'k here: {k}')

print("test")

for i in os.walk("/dataset"):
    print(i)