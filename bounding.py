
import cv2
import numpy as np

def convert_bbox_3d(min_x, min_y, min_z, size_x, size_y, size_z):
 
    min_point_x = min_x
    min_point_y = min_y
    min_point_z = min_z
    
 
    
    bbox = np.zeros(shape = (8, 3))
    #lower level
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    #upper level
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])
    
    return bbox

def project_bbox_3D_to_2D(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, append_centerpoint = True):
    
    if append_centerpoint:
        points_bbox_3D = np.concatenate([points_bbox_3D, np.zeros(shape = (1, 3))], axis = 0)
    points_bbox_2D, jacobian = cv2.projectPoints(points_bbox_3D, rotation_vector, translation_vector, camera_matrix, None)
    points_bbox_2D = np.squeeze(points_bbox_2D)
    
    return points_bbox_2D

def draw_bbox_8_2D(draw_img, bbox_8_2D, color = (0, 255, 0), thickness = 2):
    #convert bbox to int and tuple
    bbox = np.copy(bbox_8_2D).astype(np.int32)
    bbox = tuple(map(tuple, bbox))
    
    #lower level
    cv2.line(draw_img, bbox[0], bbox[1], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[2], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[3], color, thickness)
    cv2.line(draw_img, bbox[0], bbox[3], color, thickness)
    #upper level
    cv2.line(draw_img, bbox[4], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[5], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[6], bbox[7], color, thickness)
    cv2.line(draw_img, bbox[4], bbox[7], color, thickness)
    #sides
    cv2.line(draw_img, bbox[0], bbox[4], color, thickness)
    cv2.line(draw_img, bbox[1], bbox[5], color, thickness)
    cv2.line(draw_img, bbox[2], bbox[6], color, thickness)
    cv2.line(draw_img, bbox[3], bbox[7], color, thickness)
    
    #check if centerpoint is also available to draw
    if len(bbox) == 9:
        #draw centerpoint
        cv2.circle(draw_img, bbox[8], 3, color, -1)


def draw_detections(image, boxes, rotations, translations, camera_matrix):
   
    translation_vector = translations
    points_bbox_2D = project_bbox_3D_to_2D(boxes, rotations, translation_vector, camera_matrix, append_centerpoint = True)/2
    print('box',points_bbox_2D)
    draw_bbox_8_2D(image, points_bbox_2D)

def get_camera_matrix():
    607.499, 607.42, 325, 231.3
    return np.array([[607.499, 0., 325.], [0.,607.42, 231], [0., 0., 1.]], dtype = np.float32)

def drawing(image,rotation,translation):
    boxes = convert_bbox_3d( -107.83500000, -60.92790000,  -109.70500000,  215.67000000,  121.85570000,  219.41000000)
    
    draw_detections(image,boxes,rotation,translation, camera_matrix=get_camera_matrix())
    cv2.imwrite('bb.png', image)