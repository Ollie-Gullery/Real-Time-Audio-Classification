import random
import os
import tempfile
from flask import Flask, request, jsonify
from predict_model import Audio_Classifier
import hydra
from omegaconf import DictConfig, OmegaConf




# instantiting flask app 
app = Flask(__name__)


@app.route("/predict", methods=["POST"])
def process():
    audio_file = request.files["file"]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            audio_file.save(temp_file.name)
            
            # Invoke Audio Classifier
            ac = Audio_Classifier()
            
            prediction = ac.predict(temp_file.name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    # file_name = str(random.randint(0,100000))
    # audio_file.save(file_name)
    
    # # Invoke Audio Classifier 
    # ac = Audio_Classifier()
    
    # prediction = ac.predict(file_name)
    
    # # remove audio file
    # os.remove(file_name)
    
    result = {"Prediction:" : prediction}  
    
   
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)

