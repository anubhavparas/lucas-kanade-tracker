import cv2
import numpy as np 
import math
import copy

from commonutils import *
from lucaskanade import pyr_LK_Tracker


if __name__ == "__main__":
    frame = 1
    frame_str = str(frame).zfill(4) 
    folder_path = 'DragonBaby/img/'
    img_file = folder_path + frame_str + '.jpg'
    template = cv2.imread(img_file)
    height, width, _ = template.shape
    #template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    roi = np.array([[149, 63], [223, 154]])  # baby
    
    rect_tl_pt = np.array([roi[0][0], roi[0][1], 1])
    rect_br_pt = np.array([roi[1][0], roi[1][1], 1])
    frame = 2

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter("track_baby3.avi", fourcc, 10.0, (width,height))

    num_layers = 1
    template_copy = copy.deepcopy(template)
    template = get_template(template, roi, num_layers)
    threshold = 0.001

    while True:
        img_file = folder_path + frame_str + '.jpg'
        image = cv2.imread(img_file)
        
        if image is None or cv2.waitKey(1) == 27:
            print('No Image found')
            break
            
        image_copy = copy.deepcopy(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.GaussianBlur(image, (5, 5), 5)
        image = normalize_image(image, template)
        
        p = np.zeros(6)
        p_prev = p
        
        p = pyr_LK_Tracker(image, template, roi, num_layers, threshold, False)
        warp_mat = np.array([[1 + p[0], p[2], p[4]], [p[1], 1 + p[3], p[5]]])
    
        rect_tl_pt_new = (warp_mat @ rect_tl_pt).astype(int)
        rect_br_pt_new = (warp_mat @ rect_br_pt).astype(int)
  
        cv2.rectangle(image_copy, tuple(rect_tl_pt_new), tuple(rect_br_pt_new), (255, 255, 0), 1)
        cv2.imshow('Tracked Image', image_copy)
        
        frame += 1
        frame_str = str(frame).zfill(4)
        p_prev = p
        
        
        roi = update_roi_baby(frame, roi)
        rect_tl_pt = np.array([roi[0][0], roi[0][1], 1])
        rect_br_pt = np.array([roi[1][0], roi[1][1], 1])
        
        
        #print('frame----------------', frame)
        
        out.write(image_copy)
        
        
    out.release()    
    cv2.destroyAllWindows()
