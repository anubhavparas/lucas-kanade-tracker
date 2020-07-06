import numpy as np
import cv2
import math
import copy



def jacobian(x_shape, y_shape):
    # get jacobian of the template size.
    x = np.array(range(x_shape))
    y = np.array(range(y_shape))
    x, y = np.meshgrid(x, y) 
    ones = np.ones((y_shape, x_shape))
    zeros = np.zeros((y_shape, x_shape))

    row1 = np.stack((x, zeros, y, zeros, ones, zeros), axis=2)
    row2 = np.stack((zeros, x, zeros, y, zeros, ones), axis=2)
    jacob = np.stack((row1, row2), axis=2)

    return jacob


def get_template(image, roi, num_layers):
    template = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    template = cv2.GaussianBlur(template, (5, 5), 5) 
    template = resample_image(template, num_layers, resample=cv2.pyrDown)
    
    scale_down = 1/2**num_layers
    roi = (roi * scale_down).astype(int)
    
    #template = (template - np.mean(template)) / np.std(template)
    #return crop(template, roi)
    return template

def normalize_image(image, template):
    image = (image * (np.mean(template)/np.mean(image))).astype(float)
    return image



def resample_image(image, iteration, resample):
    for i in range(iteration):
        image = resample(image)
    return image



def crop(img, roi):
    return img[roi[0][1]:roi[1][1], roi[0][0]:roi[1][0]]




def gamma_correction(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def equalize_light(image, limit=12.0, grid=(2,2), gray=False):
    if (len(image.shape) == 2):
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        gray = True
    
    clahe = cv2.createCLAHE(clipLimit=limit, tileGridSize=grid)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    cl = clahe.apply(l)
    #cl = cv2.equalizeHist(l)
    limg = cv2.merge((cl,a,b))

    image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    if gray: 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return np.uint8(image)


def update_roi_bolt(frame, roi):
    roi_map = {
        50: np.array([[221, 73], [278, 173]]),
        110: np.array([[206, 53], [269, 174]]),
        150: np.array([[272, 70], [319, 166]]),
        190: np.array([[327, 66], [381, 162]]),
        220: np.array([[327, 89], [382, 172]]),
        250: np.array([[351, 98], [420, 173]]),
        280: np.array([[351, 85], [410, 176]]), 
    }
    return roi_map.get(frame, roi)



def update_roi_car(frame, roi):
    roi_map = {
        50: np.array([[64, 52], [167, 133]]),
        100: np.array([[81, 58], [163, 127]]),
        130: np.array([[82, 64], [176, 138]]),
        160: np.array([[100, 55], [199, 134]]),
        180: np.array([[116, 59], [198, 128]]),
        210: np.array([[135, 60], [227, 130]]),
        240: np.array([[160, 58], [248, 126]]),
        280: np.array([[191, 59], [261, 119]]),
        320: np.array([[200, 65], [278, 122]]),
        400: np.array([[221, 74], [295, 128]])
    }
    return roi_map.get(frame, roi)


def update_roi_baby(frame, roi):
    roi_map = {
        14: np.array([[133, 78], [207, 141]]),
        44: np.array([[21, 44], [135, 105]]),
        55: np.array([[193, 81], [259, 132]]),
        80: np.array([[94, 133], [209, 252]]),
        90: np.array([[166, 63], [253, 160]])
    }
    return roi_map.get(frame, roi)