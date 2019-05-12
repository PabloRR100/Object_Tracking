#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  9 18:06:40 2019
@author: pabloruizruiz
"""

import cv2
import time
import imutils
from imutils.video import VideoStream

'''
OpenCV already comes with many prebuilt object trackers we are going to explore
## TODO: Add more comments ...
'''

# =============
# CONFIGURATION
# =============

print('OpenCV Version ', cv2.__version__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--video", type=str, help="path to input video file")
parser.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
args = vars(parser.parse_args())

# OpenCV Object Trackers

OPENCV_OBJECT_TRACKERS = {
		"csrt": cv2.TrackerCSRT_create,
		"kcf": cv2.TrackerKCF_create,
		"boosting": cv2.TrackerBoosting_create,
		"mil": cv2.TrackerMIL_create,
		"tld": cv2.TrackerTLD_create,
		"medianflow": cv2.TrackerMedianFlow_create,
		"mosse": cv2.TrackerMOSSE_create
	}

# OpenCV Multi Object Tracker
trackers = cv2.MultiTracker_create()


# =================
# VIDEO | RECORDING
# =================


## DEFINE INPUT

# If path to video not provided, start camera
if not args.get("video", False):
	print("[INFO] starting video stream...")
	vs = VideoStream(src=0).start()
	time.sleep(1.0)

# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
    
    
## INITIALIZE TRACKER
    
fps = None      # FPS throughput estimator --> Will allows us to pause video stream
initBB = None   # We start not tracking anything

## START INFINITE LOOP

while True:
	
	frame = vs.read()
	frame = frame[1] if args.get("video", False) else frame        ## TODO: why frame[1] ?

	# Is video finished ?
	if frame is None:
		break

	## PREPOCESSING
    
    # Resize
	frame = imutils.resize(frame, width=600)
#	(H, W) = frame.shape[:2]

	# We DONT have the initBB here we are not going to select anything
    
	# Trackers return the detection status and a BB if succeded
	(detected, box) = trackers.update(frame)
    
	## CASE THERE WAS A DETECTION
	if detected:
        
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	# Show the output frame
	cv2.imshow("Frame", frame)
    
    ## KEYBOARD KEYS
    
	key = cv2.waitKey(1) & 0xFF

	# If 's' --> we are going to "select" a BB to track
	if key == ord("s"):
        
		# ENTER or SPACE after selecting the ROI
		box = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

        # Create a new tracker for the BB and add it to our multi-object tracker
		trackers.add(OPENCV_OBJECT_TRACKERS[args["tracker"]](), frame, box)
        
	# If `q` key --> Exit program
	elif key == ord("q"):
		break

# if we are using a webcam, release the pointer
if not args.get("video", False):
	vs.stop()

# otherwise, release the file pointer
else:
	vs.release()

# close all windows
cv2.destroyAllWindows()