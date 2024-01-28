import random
import os
from flask import Flask, request, jsonify, render_template, redirect
# from predict_model import Audio_Classifier
import hydra
from omegaconf import omegaconf, dictconfig
# instantiting flask app 
app = Flask(__name__)


@app.route("/predict", methods=["Get", "POST"])
def home():
    return render_template("src/frontend/templates/index.html")
    # return "Hello!"

# @app.route("/predict", methods=["POST"])
# def predict(cfg: dictconfig):
#     """Endpoint to predict keyword

#     :return (json): This endpoint returns a json file with the following format:
#         {
#             "keyword": "down"
#         }
#     """
#     # get file from POST request and save it
#     audio_file = request.files["file"]
#     file_name = str(random.randint(0, 100000))
#     audio_file.save(file_name)

#     # instantiate keyword spotting service singleton and get prediction
#     audio_clf = Audio_Classifier()
#     predicted_keyword = audio_clf.predict(file_name)

#     # we don't need the audio file any more - let's delete it!
#     os.remove(cfg.real_time.wave_output_filename)

#     # send back result as a json file
#     result = {"keyword": predicted_keyword}

#     return jsonify(result)


# @app.route("/audio_clf", methods=["POST"])
# def predict(cfg: dictconfig):
#     """Endpoint to predict keyword

#     :return (json): This endpoint returns a json file with the following format:
#         {
#             "keyword": "down"
#         }
#     """
#     # get button click and record it
    
    
#     # invoke classification model 
    
    
#     # remove ouput file (optional)
    
    



if __name__ == "__main__":
    app.run(debug=False)

