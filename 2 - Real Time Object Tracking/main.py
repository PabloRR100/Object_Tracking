
import cv2
import time
import imutils
from imutils.video import FPS
from imutils.video import VideoStream


# =============
# CONFIGURATION
# =============


## Main Config

print('OpenCV Version: ', cv2.__version__)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-v', '--video', type=str, help='path to the input video file')
parser.add_argument('-t', '--tracker', type=str,  default='kfc', help='path to the input video file')
args = vars(parser.parse_args())


## Tracker Config

OPENCV_TRACKERS = {
        'cstr': cv2.TrackerCSRT_create,
        'kfc': cv2.TrackerKCF_create,
        'boosting': cv2.TrackerBoosting_create,
        'mil': cv2.TrackerMIL_create,
        'tld': cv2.TrackerTLD_create,
        'medianflow': cv2.TrackerMedianFlow_create,
        'mosse': cv2.TrackerMOSSE_create}


# =================
# VIDEO | RECORDING
# =================


## DEFINE INPUT

# If video, start; else start camera
if not args.get('video', False):
    print('[INFO] Starting video stream...')
    vs = VideoStream(src=0).start()
    time.sleep(1.0)
    
else:
    vs = cv2.VideoCapture(args['video'])
    

## INITIALIZE TRACKER
    
tracker = OPENCV_TRACKERS[args['tracker']]()

fps = None
initBB = None

## START LOOP

while True:
    
    frame = vs.read()
    frame = frame[1] if args.get('video', False) else frame
    
    # Check if video is finished
    if frame is None:
        break
    
    ## PREPROCESSING
    frame = imutils.resize(frame, width=500)
    (H,W) = frame.shape[:2]
    
    # If we detect anything:
    if initBB is not None:
        
        # Update the tracker:
        (detected, box) = tracker.update(frame)
        
        if detected:
            
            (x,y,w,h) = [int(v) for v in box]
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2)
            
        fps.update()
        fps.stop()
        
        ## INFORMATION
        
        # Collect
        
        info = [
            ('Tracker', args['tracker']),
            ('Success', 'Yes' if detected else 'No'),
            ('FPS', '{:.2f}'.format(fps.fps()))
        ]
        
        # Draw
        for (i, (k,v)) in enumerate(info):
            text = '{}: {}'.format(k,v)
            cv2.putText(frame, text, (10, H-((i*20) +20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            
    cv2.imshow('Frame', frame)  ## TODO: Check that
    
    ## HANDLE KEYBOARD ACTIONS
    
    key = cv2.waitKey(1) & 0xFF
    
    # If 's' --> Stop FPS and Select ROI
    if key == ord('s'):
        
        # Press ENTER / SPACE BAR after selection
        initBB = cv2.selectROI('Frame', frame, fromCenter=False, showCrosshair=True)
        
        # Initialize the tracker instance
        tracker.init(frame, initBB)
        
        # Starte FPS 
        fps = FPS().start()
    
    # If 'q' --> Quit the program
    elif key == ord('q'):
        break
    

if not args.get('video', False):
    vs.stop()
    
else:
    vs.release()
    
    
cv2.destroyAllWindows()
    
        
        
    