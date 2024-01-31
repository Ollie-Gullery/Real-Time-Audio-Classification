import random
import os
from flask import Flask, request, jsonify, render_template, redirect
from predict_model import Audio_Classifier
import hydra
from audio_input import RealTimeClassification
from omegaconf import DictConfig, OmegaConf
# instantiting flask app 
app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')


@app.route("/", methods=["Get", "POST"])
def home():
    action = request.json.get('action')
    if action == 'start':
        # Code to start the action
        run_model()  # Assuming run_mode starts the action
        return jsonify({"message": "Action started"})
    elif action == 'stop':
        # Code to stop the action
        # Implement the logic to stop what run_mode() starts
        return jsonify({"message": "Action stopped"})
    return jsonify({"error": "Invalid action"}), 400
    
    return render_template("index.html")


def run_model():
        
    # instantiate audio classifier
    audio_clf = Audio_Classifier()
    # load hydra configs
    cfg = OmegaConf.load("src/config/config.yaml")
    
    # instantiate RTC 
    rtc = RealTimeClassification(audio_clf, cfg)



if __name__ == "__main__":
    app.run(debug=False)

