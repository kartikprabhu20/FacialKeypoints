from flask import Flask, render_template, Response
from camera import VideoCamera
# from display import VideoDisplay

import time
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
import cv2
from model import FacialKeypointModel

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialKeypointModel("KeyPointDetector.json", "weights.hdf5")
font = cv2.FONT_HERSHEY_SIMPLEX


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    camera.start()
    time.sleep(3.0)#warmup to get started
    
#     display = VideoDisplay(camera)

    while camera.more():
        if camera.size() < 50:
            time.sleep(0.05)
            
        fr = camera.get_frame()
        
        _, jpeg = cv2.imencode('.jpg', fr)
        frame = jpeg.tobytes()


        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
