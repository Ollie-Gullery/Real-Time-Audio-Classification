import requests

# server url
URL = "http://127.0.0.1:5050/predict"


# Audio File
TEST_AUDIO_FILE_PATH = "/Users/OliverGullery/Desktop/audio/data/raw/prediction_data/speech_predict/speech_predict.wav"

if __name__ == "__main__":
     # open files
     file = open(TEST_AUDIO_FILE_PATH, "rb")

     values = {"file": (TEST_AUDIO_FILE_PATH, file, "audio/wav")}
     response = requests.post(URL, files=values)


     data = response.json()


     print(f"Predicted keyword: {data}")
          