
import cv2
import time
import keras
import imutils

import numpy as np
from imutils.video import VideoStream

from centroid_tracker import CentroidTracker

print(cv2.__version__)
print(keras.__version__)


# CONFIGURATION 

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('-p', '--prototxt', required=True,
                    help="path to Caffe 'depoly' prototxt file")

parser.add_argument('-m', '--model', required=True,
                    help="path to Cagge pre-trained model")

parser.add_argument('-c', '--confidence', typle=float, default=0.5,
                    help="minimum probability to filter weak detections")

args = vars(parser.parse_args())


# Centroid Tracker

tracker = CentroidTracker()
H, W = None, None

# Pre-trained Deep Learning Detector
net = cv2.dnn.readNetFromCaffe(args['prototxt'], args['model'])
# net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

