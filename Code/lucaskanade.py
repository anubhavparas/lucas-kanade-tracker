import cv2
import numpy as np 
import math
import copy
from commonutils import *



def affineLKtracker(img, template, rect, p, threshold, check_brightness, max_iter=100):
        d_p_norm = np.inf
        
        template = crop(template, rect)
        rows, cols = template.shape
        
        #img = (img-np.mean(img))/np.std(img)
        p_prev = p
        iter = 0
        while (d_p_norm >= threshold) and iter <= max_iter:
            warp_mat = np.array([[1+p_prev[0], p_prev[2], p_prev[4]], [p_prev[1], 1+p_prev[3], p_prev[5]]])
           
            
            warp_img = crop(cv2.warpAffine(img, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC), rect)
            if check_brightness and np.linalg.norm(warp_img) < np.linalg.norm(template):
                #warp_img  = gamma_correction(warp_img.astype(int), gamma=1.5)
                print('inside')
                warp_img = equalize_light(warp_img.astype(int))
                
            diff = template.astype(int) - warp_img.astype(int)
        
            # Calculate warp gradient of image
            grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
            grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
            
            #warp the gradient
            grad_x_warp = crop(cv2.warpAffine(grad_x, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP), rect)
            grad_y_warp = crop(cv2.warpAffine(grad_y, warp_mat, (img.shape[1],img.shape[0]),flags=cv2.INTER_CUBIC+cv2.WARP_INVERSE_MAP), rect)
            
            # Calculate Jacobian for the 
            jacob = jacobian(cols, rows)
            
            grad = np.stack((grad_x_warp, grad_y_warp), axis=2)
            grad = np.expand_dims((grad), axis=2)
            
            #calculate steepest descent
            steepest_descents = np.matmul(grad, jacob)
            steepest_descents_trans = np.transpose(steepest_descents, (0, 1, 3, 2))
            
            # Compute Hessian matrix
            hessian_matrix = np.matmul(steepest_descents_trans, steepest_descents).sum((0,1))
         
            # Compute steepest-gradient-descent update
            diff = diff.reshape((rows, cols, 1, 1))
            update = (steepest_descents_trans * diff).sum((0,1))
            
            # calculate dp and update it
            d_p = np.matmul(np.linalg.pinv(hessian_matrix), update).reshape((-1))
                
            p_prev += d_p
            
            d_p_norm = np.linalg.norm(d_p)
            iter += 1
            
        return p_prev



def pyr_LK_Tracker(image, template, roi, num_layers, threshold, check_brightness):
    image_copy = copy.deepcopy(image)
    template_copy = copy.deepcopy(template)
    
    scale_down = 1/2**num_layers
    scale_up = 2**num_layers
    roi_down = (roi*scale_down).astype(int)
    
    image = resample_image(image, num_layers, resample=cv2.pyrDown)
    p = np.zeros(6)
    roi_pyr = roi_down
    
    #num_layers = 0  # to be removed
    for i in range(num_layers+1):
        p = affineLKtracker(image, template_copy, roi_pyr, p, threshold, check_brightness)
        
        image = resample_image(image, iteration=1, resample=cv2.pyrUp)
        template_copy = resample_image(template_copy, iteration=1, resample=cv2.pyrUp)
        roi_pyr = (roi_pyr * 2).astype(int)
        
    return p

