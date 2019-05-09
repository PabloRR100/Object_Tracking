
import cv2
import time
import imutils

import numpy as np
from imutils.video import VideoStream
from centroid_tracker import CentroidTracker


# =============
# CONFIGURATION
# =============

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--prototxt', default='./deploy.protxt')
parser.add_argument('-m', '--model', default='./res10_300x300_ssd_iter_140000.caffemodel')
parser.add_argument('-c', '--confidence', type=float, default=0.5)
args = vars(parser.parse_args())


# Load Tracker
tracker = CentroidTracker()
H,W = None, None


# Load the Model

import os
assert os.path.exists(args['prototxt']), 'Config file not found'
assert os.path.exists(args['model']), 'Model file not found'
#print(args['prototxt'])

net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])



# ===========
# START VIDEO
# ===========

video = VideoStream(src=0).start()
time.sleep(2.0)

h,w = (225, 400)
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
file = cv2.VideoWriter_('output.mov', fourcc, 20.0, (h,w))

while True:
    
    # Start Video
    frame = video.read()
    frame = imutils.resize(frame, width=400)
    
    # Preprocessing
    if W is None or H is None: (H,W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (H,W), (104., 177., 123.))
    
    # Forward Pass of our model
    net.setInput(blob)
    detections = net.forward()
    
    # Handle Detections
    rects = []
    for i in range(detections.shape[2]):
        
        # Filter week detections
        if detections[0,0,i,2] > args['confidence']:
            
            box = detections[0,0,i,3:7] * np.array([W,H,W,H])
            rects.append(box.astype('int'))
            (x0,y0,x1,y1) = box.astype('int')
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            
    objects = tracker.update(rects)
    
    for objectID, centroid in objects.items():
        
        point_loc = centroid[0], centroid[1]
        cv2.circle(frame, point_loc, 4, (0,255,0), -1)
        
        text = 'ID {}'.format(objectID)
        text_loc = centroid[0] -10, centroid[1] -10
        cv2.putText(frame, text, text_loc, cv2.FONT_HERSHEY_SIMPLE, 0.5, (0,255,0), 2)
        
    # Show the output
    file.write(frame)
    cv2.imshow('Frame', frame)
     
    # Quit
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    
file.release()
video.stop()
cv2.destroyAllWindows()