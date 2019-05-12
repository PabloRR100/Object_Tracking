
video=$1

if [ $1 = 1 ]
then
$video = nascar.mp4


echo Activating Enviroment...
source activate pytorch

echo Running File...
python main_commented --video $video
exit