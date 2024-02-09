import tkinter as tk
from audio_input import RealTimeClassification
import hydra
from omegaconf import DictConfig, OmegaConf
from deployment.classifier.predict_model import Audio_Classifier
from threading import Thread

class AudioClassifierGUI:
    def __init__(self, master):
        self.master = master
        master.title("Real-Time Audio Classifier")

        # Load configuration and create RealTimeClassification instance
        cfg = OmegaConf.load("src/config/config.yaml")
        self.rtc = RealTimeClassification(Audio_Classifier(), cfg)

        # Start Button
        self.start_btn = tk.Button(master, text="Start Classification", command=self.start_classification)
        self.start_btn.pack()

        # Stop Button
        self.stop_btn = tk.Button(master, text="Stop Classification", command=self.stop_classification, state='disabled')
        self.stop_btn.pack()

    def start_classification(self):
        self.start_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        # Start the classification process in a separate thread to keep the GUI responsive
        self.classification_thread = Thread(target=self.rtc.real_time_audio_classification)
        self.classification_thread.start()

    def stop_classification(self):
        self.rtc.stop_signal.set()
        self.classification_thread.join()  # Wait for the thread to finish
        self.start_btn.config(state='normal')
        self.stop_btn.config(state='disabled')

if __name__ == "__main__":
    root = tk.Tk()
    gui = AudioClassifierGUI(root)
    root.mainloop()
