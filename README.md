audio classification
==============================

Music Speech Classification Project. Network Architecture built with Tensorflow and Keras. Audio processing performed with librosa, audiomentations, and PyAudio. 

`MacOS`: To Run File (assumes conda is already installed):

1. Clone the repo: `git clone https://github.com/Ollie-Gullery/Real-Time-Audio-Classification.git`
2. Navigate to file: `cd Real-Time-Audio-Classification`
3. Get Run Permissions: `chmod +x setup_and_run.sh`
4. Start audio stream to classify between music & speech: `./setup_and_run.sh`


Project Organization
------------

    ├── LICENSE
    ├── **setup_and_run.sh**
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump (audio files)
    │
    ├── models             <- Trained and serialized model.keras file
    │
    ├── notebooks          <- Jupyter notebooks for minor EDA with librosa
    │                        
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── audio.yml  <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `conda env export --from-history > audio.yml` (pip installations installed separately)
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   ├── **audio_input.py**  <- Real Time Classification File with Audio Stream**
    │   │
    │   ├── config.yml    <- Contains all configurations for files (model configurations, preprocessing configs etc.)
    │   ├── prepare_data           <- Scripts to download or generate data
    │   │   └── **prepare_dataset.py**
    │   │   └── data_augmentations.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── **train_model.py**
    │   │
    │   ├── deployment (wip)         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── classifier
    │   │   │    │                 
    │   │   │    └── **predict_model.py**
    │   │   └── flask_setup
    │   │   │    │                 
    │   │   │    └── dockerfile
    │   │   │    └── server.py
    │   │   └── nginx
    │   │   │    │                 
    │   │   │    └── client.py


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>


