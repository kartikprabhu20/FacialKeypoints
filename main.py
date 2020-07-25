from flask import Flask, render_template, Response
from camera import VideoCamera
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
    print("sleep")

    time.sleep(3.0)
    print("awake")

    facialpoints_df = pd.read_csv('KeyFacialPoints.csv')
    columnNames = facialpoints_df.columns[:-1]

    while camera.more():
        if camera.qsize() < 10:
            time.sleep(0.1)
            
        fr = camera.get_frame()
        
#         gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
#         faces = facec.detectMultiScale(gray_fr, 1.3, 5)

#         for (x, y, w, h) in faces:
#             fc = gray_fr[y:y+h, x:x+w]

#             roi = cv2.resize(fc, (96, 96))
#             df_predict = pd.DataFrame(model.predict_keypoints(roi[np.newaxis, :, :, np.newaxis]), columns = columnNames)
        
#             cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            
#             xScale = fc.shape[0]/96
#             yScale = fc.shape[1]/96
#             for j in range(1,31,2):
#                 fr = cv2.drawMarker(fr,
#                               (int(x+df_predict.loc[0][j-1] * xScale), int(y+df_predict.loc[0][j]* yScale )),
#                                (0, 0, 255),
#                                markerType=cv2.MARKER_CROSS,
#                                markerSize=10,
#                                thickness=2,
#                                line_type=cv2.LINE_AA)

#                 fr = cv2.circle(fr, (df_predict.loc[0][j-1], df_predict.loc[0][j]), radius=5, color=(0, 0, 255), thickness=-1)

        _, jpeg = cv2.imencode('.jpg', fr)
        time.sleep(0.05)    
        frame = jpeg.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
