
import cv2
import time
import imutils

import numpy as np
from imutils.video import VideoStream

from centroid_tracker import CentroidTracker

print('OpenCV Version: ', cv2.__version__)


# CONFIGURATION 

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--prototxt', default='deploy.prototxt',
#parser.add_argument('-p', '--prototxt', default='res10.prototxt',
#parser.add_argument('-p', '--prototxt', default='ResNet_101_released/ResNet-101-deploy_augmentation.prototxt',                    
                    help="path to Caffe 'deploy' prototxt file")

parser.add_argument('-m', '--model', default='res10_300x300_ssd_iter_140000.caffemodel',
#parser.add_argument('-m', '--model', default='./res10_ssd.caffemodel',
#parser.add_argument('-m', '--model', default='ResNet_101_released/snap_resnet__iter_120000.caffemodel',
                    help="path to Caffe pre-trained model")

parser.add_argument('-c', '--confidence', type=float, default=0.5,
                    help="minimum probability to filter weak detections")

args = vars(parser.parse_args())


# Instantiate Centroid Tracker

print('Loading Centroid Tracker...')
tracker = CentroidTracker()
H, W = None, None


# Pre-trained Deep Learning Detector

import os
print('Config File found: ', os.path.exists(args['prototxt']))
print('Model File found: ', os.path.exists(args['model']))
print('Loading Config File... ', args['prototxt'])
print('Loading Model File... ', args['model'])
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])


# VIDEO CAPTURE

print('Opening webcam...')
video = VideoStream(src=0).start()

time.sleep(2.0)
# h,w = video.resolution
h,w = (225, 400)
print('Start recording...')
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
cam = cv2.VideoWriter('output.mov',fourcc, 20.0, (h,w))

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
    cam.write(frame)
    cv2.imshow('Frame', frame)   
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"): break
        
cam.release()
video.stop()
cv2.destroyAllWindows()
           
print('Exited')     
    
    