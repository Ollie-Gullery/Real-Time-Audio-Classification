import requests

# server url
URL = "http://127.0.0.1:5000/predict"


# audio file
FILE_PATH = "data/raw/prediction_data/speech_predict/speech_predict.wav"

if __name__ == "__main__":
     # open files
    file = open(FILE_PATH, "rb")

    # package stuff to send and perform POST request
    values = {"file": (FILE_PATH, file, "audio/wav")}
    response = requests.post(URL, files=values)
    data = response.json()

    print(f"Predicted keyword: {all_predictions})
          