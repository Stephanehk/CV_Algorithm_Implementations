import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

img = cv2.imread("/Users/2020shatgiskessell/Desktop/CV_Algorithm_Implementations/test_image_2.png")


def corner_detector (img, block_size, thresh, k):
    corners = []
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #gradient of image in x direction
    perwitt_x = np.array([[1,0,-1],[1,0,-1], [1,0,-1]])
    sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    img_dx = cv2.filter2D(img_gray, -1, sobel_x)
    #gradient of image in y direction
    perwitt_y = np.array([[1,1,1],[0,0,0], [-1,-1,-1]])
    sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    img_dy = cv2.filter2D(img_gray, -1, sobel_y)

    #sliding window
    windowSize = [block_size,block_size]
    stepSize = block_size
    guass_kernel = cv2.getGaussianKernel(block_size,1)
    for y in range(0, img_gray.shape[0], stepSize):
            for x in range(0, img_gray.shape[1], stepSize):
                roi = img_gray[y:y + windowSize[1], x:x + windowSize[0]]
                #get roi derivitives
                roi_dx = img_dx[y:y + windowSize[1], x:x + windowSize[0]]
                roi_dy = img_dy[y:y + windowSize[1], x:x + windowSize[0]]
                #get gaussian sum of roi derivitives
                xx = (roi_dx*roi_dx).sum()
                yy = (roi_dy*roi_dy).sum()
                xy =(roi_dx*roi_dy).sum()
                #find determenant and trace
                det = (xx*yy) - (np.power(xy,2))
                trace = xx+yy
                #corner coefficient
                R = det-k*(np.power(trace,2))
                if R > thresh:
                    cv2.rectangle(img, (x,y), (x + windowSize[0],y + windowSize[1]), (0,0,255), 2)
                    corners.append(roi)
    cv2.imshow("img_with_corners", img)
    cv2.waitKey(0)
    return corners
corner_detector (img, 2, 0.00001, 0.04)
