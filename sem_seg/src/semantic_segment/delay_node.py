#!/usr/bin/env python

import rospy
import time

if __name__ == '__main__':
    rospy.init_node('delay_node')
    delay = rospy.get_param('/delay')
    time.sleep(delay)
