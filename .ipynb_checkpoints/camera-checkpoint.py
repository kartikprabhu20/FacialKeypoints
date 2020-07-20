import cv2
from model import FacialKeypointModel
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialKeypointModel("KeyPointDetector.json", "weights.hdf5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture('/Users/apple/Workspace/Deeplearning/FacialKeypoints/videos/presidential_debate.mp4')
        facialpoints_df = pd.read_csv('KeyFacialPoints.csv')
        self.columns = facialpoints_df.columns[:-1]
        #For video from webcam
#         self.video = cv2.VideoCapture(0)


    def __del__(self):
        self.video.release()

    # returns camera frames along with bounding boxes and predictions
    def get_frame(self):
        _, fr = self.video.read()
        gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        faces = facec.detectMultiScale(gray_fr, 1.3, 5)

        for (x, y, w, h) in faces:
            fc = gray_fr[y:y+h, x:x+w]

            roi = cv2.resize(fc, (96, 96))
            df_predict = pd.DataFrame(model.predict_keypoints(roi[np.newaxis, :, :, np.newaxis]), columns = self.columns)
        
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
            
            xScale = fc.shape[0]/96
            yScale = fc.shape[1]/96
            for j in range(1,31,2):
                fr = cv2.drawMarker(fr,
                              (int(x+df_predict.loc[0][j-1] * xScale), int(y+df_predict.loc[0][j]* yScale )),
                               (0, 0, 255),
                               markerType=cv2.MARKER_CROSS,
                               markerSize=10,
                               thickness=2,
                               line_type=cv2.LINE_AA)
    
#                 fr = cv2.circle(fr, (df_predict.loc[0][j-1], df_predict.loc[0][j]), radius=5, color=(0, 0, 255), thickness=-1)

        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()
