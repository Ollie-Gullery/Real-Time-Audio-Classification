
import os
import sys
# from deployment.classifier.predict_model import _Audio_Classifier
import random
import tempfile
from flask import Flask, request, jsonify
import hydra
from omegaconf import DictConfig, OmegaConf
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from classifier.predict_model import Audio_Classifier


# # instantiting flask app 
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def process():
    audio_file = request.files["file"]
    
    # create temporary .wav file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_file.save(temp_file.name)
            
            # Invoke Audio Classifier
            ac = Audio_Classifier()
            
            # make prediction
            prediction = ac.predict(temp_file.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    result = {"Prediction:" : prediction}  

   
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)

