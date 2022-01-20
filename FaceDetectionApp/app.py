import os
import sys
import io
import cv2
from PIL import Image
import base64
from base64 import b64encode

import json
import time

import requests
import urllib.request

from flask import Flask, render_template, Response,  request, session, redirect, url_for, send_from_directory, flash
from werkzeug.utils import secure_filename

from camera import VideoCamera
from yolo import Model

app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.secret_key = "secret key"
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("home.html")

@app.route("/home")
def home():
    return render_template("home.html")

@app.route('/theory')
def theory():
    return render_template('theory.html')

@app.route('/training')
def training():
    return render_template('training.html')

@app.route('/demo')
def demo():
    return render_template('demo.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/image')
def upload_form():
    return render_template('upload.html')

@app.route('/image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)

        model = Model()
        img_pred = model.predict(img)

        # convert numpy array to PIL Image
        img = Image.fromarray(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))

        # create file-object in memory
        file_object = io.BytesIO()

        # write PNG in file-object
        img.save(file_object, 'PNG')

        # move to beginning of file so `send_file()` it will read from start    
        file_object.seek(0)

        base64img = "data:image/png;base64,"+b64encode(file_object.getvalue()).decode('ascii')

        return render_template('upload.html', image = base64img)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)

def gen(camera):
    while True:
        frame = camera.get_frame_model()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)