from __future__ import division, print_function
import os
import json
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from src.prediction import prediction

# Reading Model Path

with open("reports/metrics/scores.json", "r") as f:
    data = json.load(f)
MODEL_PATH = data['model_scores'][-1]['model_path']

# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    # Get the file from post request

    f = request.files['file']
    print(f.filename)

    # Save the file to ./uploads

    base_path = os.path.dirname(__file__)
    print("base path obtained")

    if 'n' in f.filename:
        file_path = os.path.join(base_path, 'data/test/NORMAL', secure_filename(f.filename))
    else:
        file_path = os.path.join(base_path, 'data/test/PNEUMONIA', secure_filename(f.filename))

    print(file_path)
    print('Model Prediction started')

    # Make prediction
    output = prediction(img_path=file_path, model_path=MODEL_PATH)

    return output


if __name__ == '__main__':
    app.run(port=5001, debug=True)
