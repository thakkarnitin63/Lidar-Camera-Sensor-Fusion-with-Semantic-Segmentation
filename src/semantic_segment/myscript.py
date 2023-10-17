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
import threading 


# Load the pre-trained model
def callback_seg(data):
    rospy.loginfo("i have subscribe to")
    # rate=rospy.Rate(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes=19
    output_stride=16
    model = network.modeling.__dict__['deeplabv3plus_mobilenet'](num_classes=num_classes, output_stride=output_stride)
# network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)
    checkpoint = torch.load('/home/nitin/sem_seg/src/semantic_segment/best_deeplabv3plus_mobilenet_cityscapes_os16.pth', map_location=torch.device('cpu'))

    model.load_state_dict(checkpoint["model_state"])
    model = nn.DataParallel(model)
    model.to(device)
    print("Resume model from")
    del checkpoint

    print("Device: %s" % device)
    with torch.no_grad():
        model = model.eval()
    frame = br.imgmsg_to_cv2(data)
    # if input_cv2 == None:
    #      print("none")
    # Start the camera capture
    # cap = cv2.VideoCapture(input_cv2)
    decode_fn = Cityscapes.decode_target
    # Main loop
        # Capture frame-by-frame
        # ret, frame = cap.read()

#         # Preprocess image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (513,513))
    image = transforms.ToTensor()(image)
    image = image.unsqueeze(0).to(device)
        # Perform inference
    with torch.no_grad():
        output = model(image).max(1)[1].cpu().numpy()[0]
        colorized_preds = decode_fn(output).astype('uint8')
        # colorized_preds = Image.fromarray(colorized_preds)
        # colorized_preds = np.array(colorized_preds)
        # colorized_preds = cv2.resize(colorized_preds, (513, 513))
        seg_img = br.cv2_to_imgmsg(colorized_preds)
        #####publish the output #####
        pub = rospy.Publisher('segmentation', Image,queue_size=1000)
        rate=rospy.Rate(10)
        rospy.loginfo("publishing the segmentation at segmentation topic")
        # print(seg_img)
        
        pub.publish(seg_img)
#     # Convert output to color map for visualization
def subscriber():
        rospy.init_node('seg_subs',anonymous=True)
        rospy.Subscriber('/camera/color/image_raw',Image,callback=callback_seg)
        rospy.spin()
        

if __name__=='__main__':
    try:
        # publish_video()
        subscriber()
        # threads = []
        # for i in range(2): # set the number of threads to 2
            # t = threading.Thread(target=subscriber)
            # threads.append(t)
            # t.start()

        # for thread in threads:
            # thread.join()
    except rospy.ROSInterruptException:
        pass





