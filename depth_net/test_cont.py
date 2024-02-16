#!/usr/bin/env python
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
import os
#import numpy as np

def convert_depth_image(ros_image):
    ###############Load CV_bridge and check if loaded##########	
    global depth_image	
    cv_bridge = CvBridge()
    try:
        depth_image = cv_bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
    except CvBridgeError as e:
        print(e)

def convert_rgb_image(ros_image):
    ###############Load CV_bridge and check if loaded##########	
    global rgb_image
    cv_bridge = CvBridge()
    
    try:
        rgb_image = cv_bridge.imgmsg_to_cv2(ros_image, "bgr8")
    except CvBridgeError as e:
        print(e)

def convert_thermal_image(ros_image):
    ###############Load CV_bridge and check if loaded##########	
    global thermal_image
    cv_bridge = CvBridge()
    
    try:
        thermal_image = cv_bridge.imgmsg_to_cv2(ros_image, 'mono16')
    except CvBridgeError as e:
        print(e)

def Grabber(imag):
    global rgb_folder, depth_folder, thermal_folder   
    if not 'i' in globals():
        global i, a
        i = 0
        a = 0

    
    if i % 10 == 0 :
        
        rgb_ = "frame_%06d.png" % a
        rgb_folder = os.path.join( '/home/mahmoud/Downloads/testing_depth/rgb', rgb_)
        depth_folder = os.path.join( '/home/mahmoud/Downloads/testing_depth/depth' , rgb_)
        thermal_folder = os.path.join( '/home/mahmoud/Downloads/testing_depth/thermal' , rgb_)
        cv2.imwrite(depth_folder, depth_image)
        cv2.imwrite(rgb_folder, rgb_image)
        cv2.imwrite(thermal_folder, thermal_image)
        print("saving rgb,depth,and thermal" " Frame: " + str(a))
        a +=1
   
    i += 1
    
    	
	

def pixel2depth():
    rospy.init_node('pixel2depth',anonymous=True)
    rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image,callback=convert_depth_image, queue_size=1)
    rospy.Subscriber("/camera/color/image_raw", Image,callback=convert_rgb_image, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw", Image,callback=convert_thermal_image, queue_size=1)
    rospy.Subscriber("/usb_cam/image_raw", Image,callback=Grabber, queue_size=1)
    rospy.spin()

if __name__ == '__main__':
	pixel2depth()

