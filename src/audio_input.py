import pyaudio
import wave
import tensorflow as tf
from predict_model import Audio_Classifier
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import librosa


# audio_clf = Audio_Classifier()


def real_time_audio_classification(audio_clf):

    
    # Initialize PyAudio for audio input
    p = pyaudio.PyAudio()
    
    # Define audio input settings
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    SAMPLE_RATE = 22050
    WINDOW_SIZE = 0.25 # 0.25seconds
    CHUNK = int(SAMPLE_RATE * WINDOW_SIZE)
    WAVE_OUTPUT_FILENAME = "output.wav"
    prediction_array = []
    frames = []
    print("Recording")
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    print("Real-time audio classification started. Press Ctrl+C to stop.")
    
    try:
        count = 0
        while True:
            data = stream.read(CHUNK)
            frames.append(data)
            signal = np.frombuffer(data, dtype=np.int16)
            
            # Normalize to float32 in range [-1, 1] so it can be processed by librosa 
            
            signal = signal.astype(np.float32) / np.iinfo(np.int16).max
            # # Perform audio classification on the incoming audio
            predicted_class = audio_clf.predict_signal(signal)
            count +=1
            print(f"Count: {count} | Predicted Class: {predicted_class}")
            prediction_array.append(predicted_class)
    except KeyboardInterrupt:
        print("Real-time audio classification stopped.")

    # Close the audio stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()
    # Save the recorded audio to a WAV file
    with wave.open(WAVE_OUTPUT_FILENAME, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b''.join(frames))

    print(f"File saved as {WAVE_OUTPUT_FILENAME}")
if __name__ == "__main__":
    audio_clf = Audio_Classifier()
    real_time_audio_classification(audio_clf)
    # print(predicted_class = )





def standard_run():
    FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
    CHANNELS = 1              # Number of audio channels (1 for mono, 2 for stereo)
    RATE = 44100              # Sample rate (samples per second)
    RECORD_SECONDS = 5        # Duration of the recording in seconds
    OUTPUT_FILENAME = "recorded_audio.wav"  # Output file name

    # Initialize PyAudio
    p = pyaudio.PyAudio()

    # Open an audio stream for recording
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording...")

    frames = []

    # Record audio for the specified duration
    for _ in range(0, int(RATE / 1024 * RECORD_SECONDS)):
        data = stream.read(1024)
        frames.append(data)

    # Stop recording
    print("Recording finished.")

    # Close the audio stream
    stream.stop_stream()
    stream.close()




def check_pyaudio_version():
    p = pyaudio.PyAudio()

    # Get PortAudio version information
    portaudio_version = pyaudio.__version__


    print("Pyaudio Version:", portaudio_version)

    # # Close PyAudio
    p.terminate()


