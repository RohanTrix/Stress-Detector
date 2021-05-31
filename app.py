from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import math
import imutils
import time
import dlib
import cv2
from cv2 import VideoWriter_fourcc, VideoWriter
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import load_model

from flask import Flask, render_template, request, jsonify, redirect, Response
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/checkstress', methods = ['GET', 'POST'])
def check_stress():
    return render_template('check.html')

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def gen():
    import os
    os.system('python process.py')
    final_vid_cap = cv2.VideoCapture('resvid.avi')
    while(final_vid_cap.isOpened()):
        _, frame = final_vid_cap.read()
        if not _:
            final_vid_cap = cv2.VideoCapture('resvid.avi')
            continue
        else:
            frame = rescale_frame(frame, percent=100)
            frame = cv2.resize(frame, (0, 0), None, 1, 1)
        
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        import time
        time.sleep(0.05)
    

@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)