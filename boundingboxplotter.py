import os
import sys 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import sys
from bounding import *
import numpy as np
from PIL import Image
from dbbox import *

img_sample = '/home/mahmoud/Downloads/Depth_Estimation_network/data_acquisition/train/rgb/mage_000034.png'
orig_image = Image.open(img_sample)
img = np.array(orig_image)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

A = 90
B = 172
C = 234
D = 231


bbox = np.array([
                [A, B],
                [C, B],
                [C, D],
                [A, D],
                ])

bbox = bbox.reshape((4, 2))
bbox1 = bboxcorrect(bbox,-5,-3)

cv2.polylines(img, [bbox1], True, (0, 255, 0), 2)
bbox2 = bboxcorrect(bbox,5,-10)
cv2.polylines(img, [bbox2], True, (0, 255, 0), 2)

cv2.line(img, bbox1[0], bbox2[0], (0, 255, 0), 2)
cv2.line(img, bbox1[1], bbox2[1], (0, 255, 0), 2)
cv2.line(img, bbox1[2], bbox2[2], (0, 255, 0), 2)
cv2.line(img, bbox1[3], bbox2[3], (0, 255, 0), 2)

cv2.imwrite("test.png", img)
