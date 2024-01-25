import pyaudio
import wave
import tensorflow as tf
from predict_model import Audio_Classifier
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import librosa
import threading
import queue
import time 
# audio_clf = Audio_Classifier()
class RealTimeClassification():
    def __init__(self, audio_clf, cfg):
        self.audio_clf = Audio_Classifier()
        self.cfg = cfg
        self.chunk = int(self.cfg.real_time.sample_rate * self.cfg.real_time.window_size)
        self.format = pyaudio.paInt16
        self.frames = []
        self.stop_signal = threading.Event()
        self.data_queue = queue.Queue()
        self.prediction_array = []
    def audio_streaming_thread(self,stream):
        print("Audio Streaming Started...")
        while not self.stop_signal.is_set():
            try:
                data = stream.read(self.chunk, exception_on_overflow= False)
                print(data)
                self.data_queue.put(data)
                self.frames.append(data)
            except Exception as e:
                print(f"Streaming Thread Error: {e}")
                break
    def audio_processing_thread(self):
        
        while not self.stop_signal.is_set() or not self.data_queue.empty():
            print("Classifying...")
            data = self.data_queue.get()
            try:
                signal = np.frombuffer(data, dtype=np.int16)
                # # Normalize to float32 in range [-1, 1] so it can be processed by librosa 
                signal = signal.astype(np.float32) / np.iinfo(np.int16).max
                
                predicted_class = self.audio_clf.predict_signal(signal)
                self.prediction_array.append(predicted_class)
                print(f'Predicted Class: {predicted_class}')
            except Exception as e:
                print(f"Error Processing: {e}")
    
    def real_time_audio_classification(self):
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                channels=self.cfg.real_time.channels,
                rate=self.cfg.real_time.sample_rate ,
                input=True,
                frames_per_buffer=self.chunk)
        print("Real-Time Audio Classification started. Press ctrl+c to stop")
        
        threading.Thread(target=self.audio_streaming_thread, args=(stream,)).start()
        threading.Thread(target=self.audio_processing_thread).start()
        
        # try:
        #     while True:
        #         pass
        # except KeyboardInterrupt:
        #     print("Recording Stopped")
        #     self.stop_signal.set()
        print("Recording")
        time.sleep(3)
        print("Ending Recording")
        # End recording
        stream.stop_stream()
        stream.close()
        p.terminate()
        
        # Save the recorded audio to a WAV file
        with wave.open(self.cfg.real_time.wave_output_filename, 'wb') as wf:
            wf.setnchannels(self.cfg.real_time.channels)
            wf.setsampwidth(p.get_sample_size(self.format))
            wf.setframerate(self.cfg.real_time.sample_rate)
            wf.writeframes(b''.join(self.frames))
          
        if self.data_queue.empty(): 
            with open(self.cfg.real_time.prediction_csv, 'w') as f:
                for prediction in self.prediction_array:
                    print(prediction)
                    f.write(f"{prediction}\n")
        
        print(f"File saved as {self.cfg.real_time.wave_output_filename}")

if __name__ == "__main__":
    audio_clf = Audio_Classifier()  # Make sure to define Audio_Classifier or import it
    cfg = OmegaConf.load("src/config/config.yaml")
    rtc = RealTimeClassification(audio_clf, cfg)
    rtc.real_time_audio_classification()




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