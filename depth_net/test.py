#!/usr/bin/env python
#import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
#sys.path.append('/opt/ros/kinetic/lib/python2.7/dist-packages') # append back in order to import rospy
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from std_msgs.msg import Int8
import numpy as np



#import keyboard 
#from pynput.keyboard import Key, Controller


#i = 0
#a = str(i) 
#rgb_ = '/image_0000' + a + '.png'
#rgb_folder = '/home/axel/testing_depth/rgb'+ rgb_
#depth_folder = '/home/axel/testing_depth/depth' + rgb_



def convert_depth_image(ros_image):
	
	
	
	###############Load CV_bridge and check if loaded##########	
	global depth_image	
	cv_bridge = CvBridge()
	try:
	    depth_image = cv_bridge.imgmsg_to_cv2(ros_image, desired_encoding='passthrough')
	except CvBridgeError as e:
		print(e)

	############ Collect Depth Array from depth image and save(optional) ###############
	depth_array = np.array(depth_image, dtype=np.float32)
	np.save("depth_img.npy", depth_array)
	#rospy.loginfo(depth_array)

	#To save image as png
	# Apply colormap on depth image (image must be converted to 8-bit per pixel first)
	depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_BONE)
	#cv2.imwrite("depth_img1.png", depth_colormap)
        
	#cv2.imwrite(depth_folder, depth_image)
	#Or you use 
	#depth_array2 = depth_array.astype(np.uint16)
	#cv2.imwrite("depth_img_array.png", depth_array2)

def convert_rgb_image(ros_image):
	###############Load CV_bridge and check if loaded##########	
	global rgb_image
	cv_bridge = CvBridge()
	
	try:
	    rgb_image = cv_bridge.imgmsg_to_cv2(ros_image, "bgr8")
	except CvBridgeError as e:
	    print(e)
	
	#cv2.imwrite(rgb_folder, rgb_image)

def convert_thermal_image(ros_image):
	###############Load CV_bridge and check if loaded##########	
	global thermal_image
	cv_bridge = CvBridge()
	
	try:
	    thermal_image = cv_bridge.imgmsg_to_cv2(ros_image, "bgr8")
	except CvBridgeError as e:
	    print(e)

def Key_is_pressed(key):
    global rgb_folder, depth_folder    
    if not 'i' in globals():
       global i 
       i = 0         
    print('here')
    if key.data == 97:
        a = str(i) 
        rgb_ = '/image_0000' + a + '.png'
        rgb_folder = '/home/mahmoud/Downloads/testing_depth/rgb'+ rgb_
        depth_folder = '/home/mahmoud/Downloads/testing_depth/depth' + rgb_
		
		thermal_folder = '/home/mahmoud/Downloads/testing_depth/thermal' + rgb_
        cv2.imwrite(depth_folder, depth_image)
        cv2.imwrite(rgb_folder, rgb_image)
        print("saving depth and rgb")
        i += 1
        
     
    	
	

def pixel2depth():
	rospy.init_node('pixel2depth',anonymous=True)
	rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image,callback=convert_depth_image, queue_size=1)
	rospy.Subscriber("/camera/color/image_raw", Image,callback=convert_rgb_image, queue_size=1)
	rospy.Subscriber("/key", Int8,callback=Key_is_pressed, queue_size=1)
	rospy.spin()

if __name__ == '__main__':
	pixel2depth()
