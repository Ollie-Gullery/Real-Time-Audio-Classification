Real Time Audio Classification
==============================

Music Speech Classification Project. Network Architecture built with Tensorflow and Keras. Audio processing performed with librosa, audiomentations, and PyAudio. 

`MacOS`: To Run File (assumes conda is already installed):

1. Clone the repo: `git clone https://github.com/Ollie-Gullery/Real-Time-Audio-Classification.git`
2. Navigate to file: `cd Real-Time-Audio-Classification`
3. Get Run Permissions: `chmod +x setup_and_run.sh`
4. Start audio stream to classify between music & speech: `./setup_and_run.sh`

*Brief Project Overview*

## Project Summary
This project is designed to differentiate between two specific types of audio in real-time. It utilizes audio datasets sourced from Kaggle, with data preparation and processing conducted through the script located at *(src/prepare_data/prepare_data.py)*. The training was performed using a Convolutional Neural Network (CNN) as outlined in *(src/models/train_model.py)*, achieving a **test accuracy of 92%**. For real-time audio streaming and processing, PyAudio was employed, enabling the model to generate predictions based on segments of audio data lasting 0.25 seconds, equivalent to a sampling rate of 5500 Hz.

*Note: The configurations of the project are all listed in `src/config/config.yaml`, it contains model configurations as well as preprocessing configurations such as sample rate, resolution of frequency domain, epochs, hop length etc., configurations were performed using `Hydra` to create hierarchial configurations* 

### Data Preparation

- Data was prepared using a sliding window, ie. segmenting it into 0.25-second intervals before converting these segments into Mel Frequency Cepstral Coefficients (MFCCs (compact representation of audio signal)) with `librosa`; which were then saved in a JSON file. 
- *Model was trained on these 0.25 segments* to obtain 'real-time' classification effect. 
- To refine the model's adaptability, dataset variability was enhanced using the `audiomentations` library, improving the model's ability to generalize across varied audio inputs.

### Training (Model)

- Constructed using `Tensorflow` and `Keras`, the model is specifically designed for binary classification challenges. Thus, I employed *Binary Cross Entropy (BCE) with logits* for its loss function to finely tune this focus.
- A key objective was to *reduce latency whilst minimizing performance loss*. Thus, my model architecture features three convolutional layers, complemented by batch normalization and max pooling, culminating in a dense layer before producing the output. To curb overfitting, L2 regularization was integrated. This architecture effectively balanced accuracy with computational efficiency making it well suited for real-time classification tasks.

### Performing Real Time Classifiation
*Note: This is done through the src/audio_input.py file*

- Real-time classification was performed with streaming via PyAudio.
- To manage storing and processing the streamed audio chunks efficiently, I utilized **threading**, allowing for the simultaneous storage and processing of these chunks. A *queue system was implemented to hold the processed audio chunks*, which were then dequeued once the model generated a prediction based of the queued item.



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
    │   ├── **audio_input.py**  <- Real Time Classification File with Audio Stream
    │   │
    │   ├── config/config.yml    <- Contains all configurations for files (model configurations, preprocessing configs etc.)
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


