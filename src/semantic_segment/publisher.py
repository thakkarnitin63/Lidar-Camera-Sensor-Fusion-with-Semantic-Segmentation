#!/usr/bin/env python3
import cv2
from PIL import Image as img
import torch
import numpy as np
import network
import utils
import torch.nn as nn
import os
from sensor_msgs.msg import Image 
import torchvision.transforms as transforms
from datasets import Cityscapes, cityscapes
import rospy
from cv_bridge import CvBridge
br = CvBridge()


def publish_video():
    pub=rospy.Publisher('/camera/color/image_raw',Image,queue_size=100) 
    rospy.init_node('video_publish_node',anonymous=True)
    rate=rospy.Rate(10)
    cap=cv2.VideoCapture('/home/nitin/sem_seg/src/semantic_segment/driveseg_sample.mp4')
    while not rospy.is_shutdown():
        ret,frame=cap.read()
        if ret==True:
            rospy.loginfo("publishing video frame begins")
            pub.publish(br.cv2_to_imgmsg(frame,"bgr8"))
            rate.sleep()
        else:
            rospy.loginfo("end of video")
            break
        
        # rate.sleep()
        # rospy.spin()


if __name__=='__main__':
    # rospy.init_node("semantic_segment",anonymous=True)
    # segment()
    try:
        publish_video()
        # subscriber()
    except rospy.ROSInterruptException:
        pass