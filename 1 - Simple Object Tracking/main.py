
import cv2
import time
import imutils

import numpy as np
from imutils.video import VideoStream

from centroid_tracker import CentroidTracker

print(cv2.__version__)


# CONFIGURATION 

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--prototxt', required=True,
                    help="path to Caffe 'depoly' prototxt file")

parser.add_argument('-m', '--model', required=True,
                    help="path to Cagge pre-trained model")

parser.add_argument('-c', '--confidence', type=float, default=0.5,
                    help="minimum probability to filter weak detections")

args = vars(parser.parse_args())


# Instantiate Centroid Tracker

print('Loading Centroid Tracker...')
tracker = CentroidTracker()
H, W = None, None


# Pre-trained Deep Learning Detector

print('Loading Model...')
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
# net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')


# VIDEO CAPTURE

print('Opening webcam... ')
video = VideoStream(src=0).start()
time.sleep(2.0)

while True:
    
    # 1 - Start video
    frame = video.read()
    frame = imutils.resize(frame, width=400)
    
    # 1.2 - Grap frame dimensions
    if W is None or H is None: (H,W) = frame.shape[:2]

    # 1.3 - Preprocessing the image - create a blob
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W,H), (104., 177., 123.))
    
    # 1.4 - Forward pass on the model  
    net.setInput(blob)
    detections = net.forward()
    
    # 2 - Loop over the detections
    
    rects = []
    
    for i in range(detections.shape[2]):
        
        # 2.1 - Filter weak detections
        if detections[0,0,i,2] > args['confidence']:
            
            # 2.2 - Compute (x,y) coordinates of the bounding box
            box = detections[0,0,i,3:7] * np.array([W,H,W,H])
            rects.append(box.astype('int'))
            
            # 2.3 - Draw the bounding box
            (x0,y0,x1,y1) = box.astype('int')
            cv2.rectangle(frame, (x0,y0), (x1,y1), (0,255,0), 2)
            
    # 3 - Pass the detections to our centroid tracker
    
    objects = tracker.update(rects)
    
    # 3.1 - Loop over the tacked objects
    for objectID, centroid in objects.items():
        
        # 3.2 - Draw the ID and the centroid point
        point_loc = centroid[0], centroid[1]
        cv2.circle(frame, point_loc, 4, (0,255,0), -1)
        
        text = 'ID {}'.format(objectID)
        text_loc = centroid[0] - 10, centroid[1] - 10
        cv2.putText(frame, text, text_loc, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
        
    # 4 - Show output frame and wait to quit
    cv2.imshow('Frame', frame)   
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
        
cv2.destroyAllWindows()
video.stop()
           
print('Exited')     
    
    