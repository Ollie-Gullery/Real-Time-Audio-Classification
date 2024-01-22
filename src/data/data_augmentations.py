import audiomentations as AA
import soundfile as sf
import librosa

aug_1 = AA.Compose([
    AA.AddGaussianNoise(
        min_amplitude=0.1, max_amplitude=0.2, p =0.75
        ), 
    AA.PitchShift(
        min_semitones=-8, max_semitones=8, p =0.5
        ),
    AA.HighPassFilter(min_cutoff_freq=50, max_cutoff_freq=300, p = 0.25)
    
])

aug_2 = AA.Compose([
    AA.AddBackgroundNoise(sounds_path="data/raw/augmentation_data/static_background_noise.wav", min_snr_in_db=0, max_snr_in_db=3, p=0.5),
    
    # Apply time stretching to change the speed of audio
    AA.TimeStretch(min_rate=0.8, max_rate=1.25, leave_length_unchanged=False, p=0.5),
    
    # Apply pitch shifting to change the pitch of audio
    AA.PitchShift(min_semitones=-2, max_semitones=2, p=0.3),
    
    # Add random short noise bursts to simulate background noise 
    AA.AddShortNoises(sounds_path="data/raw/augmentation_data/short_noise.wav", min_time_between_sounds=2.0, max_time_between_sounds=8, p=1),
    
    # time masking to simulate audio cutting out 
    AA.TimeMask(min_band_part=0.1, max_band_part=0.15, fade=True, p=0.5),
    
    
    
])


# Test Augmented Audio 
file = "data/raw/dataset/speech_wav/voice.wav"
def test_augmented_audio(file, augmentation):
    signal, sr = librosa.load(file)
    
    augmented_signal = augmentation(signal, sample_rate = sr)
    sf.write("data/raw/augmentation_data/augmented_test.wav", augmented_signal, sr)
    print("Audio Augmentation Complete")
    return

if __name__ == "__main__":
    test_augmented_audio(file, aug_2)


