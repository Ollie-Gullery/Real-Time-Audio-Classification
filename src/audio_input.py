import pyaudio
import wave
import tensorflow as tf
from deployment.classifier.predict_model import Audio_Classifier
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import librosa
import threading
import queue
import time 
import os
import audioop


portaudio_text = pyaudio.get_portaudio_version_text()

print(f"PortAudio version (text): {portaudio_text}")

class RealTimeClassification():
    def __init__(self, audio_clf, cfg):
        self.audio_clf = Audio_Classifier() # create instance of audio classifier
        self.cfg = cfg 
        self.chunk = int(self.cfg.real_time.sample_rate * self.cfg.real_time.window_size) # creat chunk of appropriate size
        self.format = pyaudio.paInt16
        self.frames = []
        self.stop_signal = threading.Event()
        self.data_queue = queue.Queue()
        self.prediction_array = []
        
        
    @staticmethod
    def save_single_queue_item_to_wav(data, file_path, channels, sample_width, frame_rate):
        """_summary_
        Saves Data Audio Object created with pyaudio to .wav file so it can be processed with librosa
        Args:
        data (bytes): The binary audio data read from the PyAudio stream. This is the raw audio data.
        file_path (str): The path where the .wav file will be saved. 
        channels (int): The number of audio channels. 
        sample_width (int): The sample width of the audio data in bytes. This is determined by the PyAudio stream's format 
        frame_rate (int): The frame rate or sample rate of the audio data. This is the number of samples collected per second 
        """
        with wave.open(file_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(frame_rate)
            wf.writeframes(data)
    
    def audio_streaming_thread(self,stream):
        """_summary_
        Obtains data from pyaudio stream of size "chunk" and appends it to a queue to be processed
        Args:   
            stream (_type_): pyaudio stream
        """
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
        """_summary_
        Processes data from Data Queue with librosa and predicts it using audio_clf class predict method
        """
        count = 0
        p = pyaudio.PyAudio()
        silence_threshold = self.cfg.real_time.silence_threshold
        while not self.stop_signal.is_set() or not self.data_queue.empty():
            print("Classifying...")
            data = self.data_queue.get()
            file_path = f'output_wav/output_{count}.wav'
            try:
                self.save_single_queue_item_to_wav(data, file_path, self.cfg.real_time.channels, p.get_sample_size(self.format), self.cfg.real_time.sample_rate)
                count += 1
                rms = audioop.rms(data, 2)
                if rms > silence_threshold:
                    predicted_class = self.audio_clf.predict(file_path)
                else:
                    predicted_class = "no noise detected"
                self.prediction_array.append(predicted_class)
                print(f'Predicted Class: {predicted_class}')
                os.remove(file_path)
            except Exception as e:
                print(f"Error Processing: {e}")
    
    def real_time_audio_classification(self):
        """_summary_
        Starts the stream and calls audio streaming and processing thread
        """
        p = pyaudio.PyAudio()
        stream = p.open(format=self.format,
                channels=self.cfg.real_time.channels,
                rate=self.cfg.real_time.sample_rate ,
                input=True,
                frames_per_buffer=self.chunk)
        print("Real-Time Audio Classification started. Press ctrl+c to stop")
        
        streaming_thread = threading.Thread(target=self.audio_streaming_thread, args=(stream,))
        processing_thread = threading.Thread(target=self.audio_processing_thread)
        streaming_thread .start()
        processing_thread.start()
        
        try:
            while True:
                time.sleep(0.1) # short sleep to reduce cpu usage
        except KeyboardInterrupt:
            print("Recording Stopped")
            self.stop_signal.set()
        # print("Recording")
        # time.sleep(15)
        # print("Ending Recording")
        # End recording
        streaming_thread.join()
        processing_thread.join()
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




