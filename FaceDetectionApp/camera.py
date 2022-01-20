import time
import numpy as np
import cv2
import pickle
import torch

from imutils.video import WebcamVideoStream
from yolo import Model

class VideoCamera(object):
    def __init__(self):
        
        # to start the webcam
        self.stream = WebcamVideoStream(0).start()
        print("stream :", self.stream)
        # Model
        self.model = Model()

    def __del__(self):
        self.stream.stop()
        # to stop the webcam
        
    def get_frame(self):
        image = self.stream.read()
        
        ext, jpeg = cv2.imencode('.jpg', image)
        
        data = []
        data.append(jpeg.tobytes())
        # this will collect images from the users through its camera
        
        return data
        # data will be sent to the Flask application
        
    def get_frame_model(self):
        image = self.stream.read()

        img_pred = self.model.predict(image)

        ret, jpeg = cv2.imencode('.jpg', img_pred)

        return jpeg.tobytes()