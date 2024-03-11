#!/bin/bash

# Define the environment name
ENV_NAME=audio


cd audio

# Create a Conda environment with required packages
conda env create -f audio.yml

# Activate the environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Installing additional dependencies
# install audiomentations
python -m pip install audiomentations

# Correctly install pyaudio with portaudio
brew install portaudio

# Automatically find the directory where portaudio.h is located
PORTAUDIO_PATH=$(find /usr/local/Cellar/portaudio/ -name "portaudio.h" | head -n 1)
PORTAUDIO_PATH=${PORTAUDIO_PATH%/*}
PORTAUDIO_LIB=${PORTAUDIO_PATH%/*}/lib

echo "Found portaudio.h in: $PORTAUDIO_PATH"
echo "Library path: $PORTAUDIO_LIB"

# Check if the paths are not empty
if [[ -z "$PORTAUDIO_PATH" || -z "$PORTAUDIO_LIB" ]]; then
    echo "Error: Could not find portaudio paths."
    exit 1
fi

# Install PyAudio specifying the include directory and library directory based on the found portaudio paths
python -m pip install --global-option='build_ext' \
                      --global-option="-I$PORTAUDIO_PATH" \
                      --global-option="-L$PORTAUDIO_LIB" \
                      pyaudio

# starting the stream 
python src/audio_input.py


