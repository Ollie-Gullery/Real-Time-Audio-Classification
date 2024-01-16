import numpy as np
import librosa 
import soundfile as sf
import matplotlib.pyplot as plt
import librosa.display
import random
# Adding white noise

class AudioDataAugmentation():
    # def __init__(self, cfg):
    #     self.cfg = cfg
    
    def plot_signal_and_augmented_signal(self, signal, augmented_signal, sr):
        fix, ax = plt.subplots(nrows=2)
        librosa.display.waveshow(signal,sr=sr,ax=ax[0], color="blue")
        ax[0].set(title="original signal")
        librosa.display.waveshow(augmented_signal, sr=sr, ax=ax[1],color="blue")
        ax[1].set(title="augmented signal")
        plt.show()
       
    
    
    # Adding white noise
    def add_white_noise(self, signal, noise_factor):
        noise = np.random.normal(0, signal.std(), signal.size)
        augmented_signal = signal + noise * noise_factor
        return augmented_signal
    
    

    # time strech
    def time_strech(signal, strech_rate):
        return librosa.effects.time_strech(signal, strech_rate )
    
    # pitch scaling
    def pitch_scale(signal, sr, num_semitones):
        return librosa.effects.pitch_shift(signal, sr, num_semitones)
    
    
    # polarity inversion
    def invert_polarity(signal):
        return signal * -1 
    
    # random gain 
    def random_gain(signal, min_gain_factor, max_gain_factor):
        gain_factor = random.uniform(min_gain_factor, max_gain_factor)
        return signal * gain_factor 
    
    
if __name__ == "__main__":
    signal, sr = librosa.load("data/raw/dataset/music_wav/bagpipe.wav")
    
    adg = AudioDataAugmentation()
    augmented_signal = adg.add_white_noise(signal, 0.5)
    
    sf.write("augmented.wav", augmented_signal, sr)
    
    adg.plot_signal_and_augmented_signal(signal, augmented_signal, sr)
    
    
    
    