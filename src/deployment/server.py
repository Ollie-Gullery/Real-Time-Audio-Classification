import random
import os
from flask import Flask, request, jsonify
from predict_model import Audio_Classifier

# instantiting flask app 
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint to predict keyword

    :return (json): This endpoint returns a json file with the following format:
        {
            "keyword": "down"
        }
    """
    # get file from POST request and save it
    audio_file = request.files["file"]
    file_name = str(random.randint(0, 100000))
    audio_file.save(file_name)

    # instantiate keyword spotting service singleton and get prediction
    audio_clf = Audio_Classifier()
    predicted_keyword = audio_clf.predict(file_name)

    # we don't need the audio file any more - let's delete it!
    os.remove(file_name)

    # send back result as a json file
    result = {"keyword": predicted_keyword}

    return jsonify(result)



if __name__ == "__main__":
    app.run(debug=False)

