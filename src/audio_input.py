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
        
    @staticmethod
    def save_single_queue_item_to_wav(data, file_path, channels, sample_width, frame_rate):
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(frame_rate)
            wf.writeframes(data)
    def audio_streaming_thread(self,stream):
        print("Audio Streaming Started...")
        while not self.stop_signal.is_set():
            try:
                data = stream.read(self.chunk, exception_on_overflow= False)
                self.data_queue.put(data)
                self.frames.append(data)
            except Exception as e:
                print(f"Streaming Thread Error: {e}")
                break
    def audio_processing_thread(self):
        count = 0
        p = pyaudio.PyAudio()
        while not self.stop_signal.is_set() or not self.data_queue.empty():
            print("Classifying...")
            data = self.data_queue.get()

            try:

                signal = np.frombuffer(data, dtype=np.int16)
                signal = signal.astype(np.float32) / np.iinfo(np.int16).max
                signal = signal[:self.chunk]
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
        time.sleep(15)
        print("Ending Recording")
        self.stop_signal.set()
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




