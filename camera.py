import cv2
from model import FacialKeypointModel
import numpy as np
import pandas as pd
from matplotlib.patches import Circle
from queue import Queue
from threading import Thread

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialKeypointModel("KeyPointDetector.json", "weights.hdf5")
font = cv2.FONT_HERSHEY_SIMPLEX

class VideoCamera(object):
    def __init__(self,queueSize=128):
        self.video = cv2.VideoCapture('/Users/apple/Workspace/Deeplearning/FacialKeypoints/videos/presidential_debate.mp4')
         #For video from webcam
#         self.video = cv2.VideoCapture(0)

        facialpoints_df = pd.read_csv('KeyFacialPoints.csv')
        self.columnNames = facialpoints_df.columns[:-1]
        
        #Flag for thread stopping
        self.stopped = False 
        #Queue to store frames
        self.Q = Queue(maxsize=queueSize)
        
        self.thread= Thread(target=self.update, args=())
        self.thread.daemon = True
        
    
    def start(self):
		# start a thread to read frames from the file video stream
        
        self.thread.start()
        return self
    
    def update(self):
		# keep looping infinitely
        count = 0
        while True:
            count +=1
            print("updating")
            
            if count%3==0:
                self.Q.put(fr)
                continue
                
            if self.stopped:
                return

            if not self.Q.full():
				# read the next frame from the file
                (grabbed, fr) = self.video.read() 

				# if the `grabbed` boolean is `False`, then we have reached the end of the video file
                if not grabbed:
                    print("stopped")
                    self.stop()
                    return
                
                gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
                
                faces = facec.detectMultiScale(gray_fr, 1.3, 5)

                for (x, y, w, h) in faces:
                    fc = gray_fr[y:y+h, x:x+w]

                    roi = cv2.resize(fc, (96, 96))
                    df_predict = pd.DataFrame(model.predict_keypoints(roi[np.newaxis, :, :, np.newaxis]), columns = self.columnNames)
                        
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
                
				# add the frame to the queue
                self.Q.put(fr)
                
    def __del__(self):
        self.video.release()
        
    def more(self):
        print("size:")
        print( self.Q.qsize())
		# return True if there are still frames in the queue
        return self.Q.qsize()>0
    
    def size(self):
		# return True if there are still frames in the queue
        return self.Q.qsize()
    
    def stop(self):
		# indicate that the thread should be stopped
        self.stopped = True
        
    # returns camera frames along with bounding boxes and predictions
    def get_frame(self): 
        return self.Q.get()
