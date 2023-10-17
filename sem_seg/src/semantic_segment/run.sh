#!/bin/bash

# sudo apt-get install terminator
# gnome-terminal --tab --command "roscore"
gnome-terminal --tab --command "rosrun semantic_segment myscript.py"
# terminator -e "rosrun semantic_segment myscript.py"
sleep 10

roslaunch lidar_camera_fusion vlp16OnImg_offline.launch