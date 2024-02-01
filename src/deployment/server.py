import random
import os
from flask import Flask, request, jsonify, render_template, redirect
from predict_model import Audio_Classifier
import hydra
# from audio_input import RealTimeClassification
from omegaconf import DictConfig, OmegaConf
from datetime import datetime
import queue
import soundfile 
from werkzeug.utils import secure_filename
# instantiting flask app 
app = Flask(__name__, static_folder='../frontend/static', template_folder='../frontend/templates')


# @app.route("/", methods=["Get", "POST"])
# def home():

    
#     return render_template("index.html")


# def run_model():
        
#     # instantiate audio classifier
#     audio_clf = Audio_Classifier()
#     # load hydra configs
#     cfg = OmegaConf.load("src/config/config.yaml")
    
#     # instantiate RTC 
#     rtc = RealTimeClassification(audio_clf, cfg)


@app.route("/predict", methods=["POST"])
def process():
    audio_file = request.files["file"]
    # timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    # filename = f"output_wav/temp_audio_{timestamp}.wav"
    file_name = str(random.randint(0,100000))
    audio_file.save(file_name)
    
    # Invoke Audio Classifier 
    ac = Audio_Classifier()
    
    prediction = ac.predict(file_name)
    
    # remove audio file
    os.remove(file_name)
    
    result = {"Prediction:" : prediction}  
    
   
    return jsonify(result)



# @app.route("/process_chunk", methods=["Get", "POST"])
# def process():

#     ac = Audio_Classifier()
    
#     if 'audio_data' not in request.files:
#         app.logger.error("No audio file in request")
#         return jsonify({"error": "No audio file"}), 400

#     try:
#         print("processing audio")
#         file = request.files['audio_data']
#         timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
#         wav_filename = f"output_wav/temp_audio_{timestamp}.wav"
#         file.save(wav_filename)
#         app.logger.info(f'Saved file {filename}')

#         result = ac.predict(file)
#     except Exception as e:
#         app.logger.error(f'Error Saving File: {e}')
#         return jsonify({"error": str(e)}), 500
#     print(result)
#     # os.remove(filename)
    
#     return jsonify({"result": result})

if __name__ == "__main__":
    app.run(debug=False)

