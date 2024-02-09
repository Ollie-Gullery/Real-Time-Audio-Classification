# Pull tensorflow image with Python3
FROM tensorflow/tensorflow:2.1.0-py3

# Set the working directory to /app
WORKDIR /app

# Transfer content from current dir to /app in container
ADD . /app

# Install audio libraries
RUN apt-get update && apt-get install -y libsndfile1 libasound-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg

# Install Conda
RUN apt-get install -y curl
RUN curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
RUN rm Miniconda3-latest-Linux-x86_64.sh

# Create a Conda environment from audio.yml
COPY audio.yml /app
RUN /opt/conda/bin/conda env create -f audio.yml

# Activate the Conda environment
RUN echo "source activate $(head -1 /app/audio.yml | cut -d' ' -f2)" > ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

# Start uWSGI using config file
CMD ["uwsgi", "app.ini"]