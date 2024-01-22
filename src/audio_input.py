import pyaudio
# import wave
# import sys

# from scipy.io.wavfile import write
# import numpy as np
# import speech_recognition as sr


import sounddevice as sd

p = pyaudio.PyAudio()

# Get PortAudio version information
portaudio_version = pyaudio.__version__


print("PortAudio Version:", portaudio_version)

# # Close PyAudio
p.terminate()

import pyaudio
import wave


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

# Terminate PyAudio
p.terminate()

# Save the recorded audio to a WAV file
with wave.open(OUTPUT_FILENAME, 'wb') as wf:
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))

print(f"Audio saved as {OUTPUT_FILENAME}")

