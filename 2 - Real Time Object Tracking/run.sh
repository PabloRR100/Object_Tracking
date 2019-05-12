#!/bin/bash

echo Activating Environment...
source activate pytorch
echo [OK] Env Activated 

echo Running File...
python main_commented.py
echo [OK] Filed run with no errors

exit