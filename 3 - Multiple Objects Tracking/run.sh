#!/bin/bash

echo Activating Enviroment...
source activate pytorch

$video = "nascar.mp4"

echo Running File...
python main_commented.py --video $video

exit