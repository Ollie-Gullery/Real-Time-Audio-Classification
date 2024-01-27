import pyaudio
import audioop
import matplotlib.pyplot as plt
import numpy as np
# Constants
FORMAT = pyaudio.paInt16  # Audio format (16-bit PCM)
CHANNELS = 1              # Number of audio channels (1 for mono)
RATE = 44100              # Sampling rate
CHUNK = 1024              # Number of frames per buffer
rms_arr = []
# Initialize PyAudio
p = pyaudio.PyAudio()

# Open stream
stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Recording. Speak into the microphone. Press Ctrl+C to stop.")

try:
    while True:
        # Read data from the stream
        data = stream.read(CHUNK, exception_on_overflow=False)

        # Calculate RMS value
        rms = audioop.rms(data, 2)
        rms_arr.append(rms)
        print(f'RMS: {rms}')

except KeyboardInterrupt:
    plt.figure(figsize=(10, 4))
    plt.plot(rms_arr, label='RMS over time')
    plt.xlabel('Frames')
    plt.ylabel('RMS Value')
    plt.title('RMS Value Over Time')
    plt.legend()
    plt.show()
    print("\nFinished recording.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()

    # Terminate the PyAudio object
    p.terminate()

print(f"max: {max(rms_arr)}, average:{np.mean(rms_arr)}, min:{min(rms_arr)}")