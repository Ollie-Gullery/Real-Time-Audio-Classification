import requests

# server url
URL = "http://127.0.0.1:5000/predict"


# audio file
TEST_AUDIO_FILE_PATH = "/Users/OliverGullery/Desktop/audio/data/raw/prediction_data/music_predict/music_predict.wav"

if __name__ == "__main__":
     # open files
     file = open(TEST_AUDIO_FILE_PATH, "rb")

     try:
          values = {"file": (TEST_AUDIO_FILE_PATH, file, "audio/wav")}
          response = requests.post(URL, files=values)
     except Exception as e:
          print(f"Error getting response: {e}")

     try:
          data = response.json()
     except Exception as e:
          print(f'Error Converting data to json format: {e}')

     print(f"Predicted keyword: {data}")
          