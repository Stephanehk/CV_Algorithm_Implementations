import numpy as np
import cv2

img = cv2.imread("/Users/2020shatgiskessell/Downloads/test_img.png")
roi = img[200:600, 800:1200]
#img[200:210, 800:810]
#img[200:600, 800:1200]
#roi = img[200:210, 800:810]
scale_factor = 2
#isolate each image channel

b, g, r = cv2.split(img)
b = roi[...,0]
g = roi[...,1]
r = roi[...,2]
#calculate Kronecker product for each channel
zoomed_no_int_b = np.kron(b, [[1,0],[0,0]])
zoomed_no_int_g = np.kron(g, [[1,0],[0,0]])
zoomed_no_int_r = np.kron(r, [[1,0],[0,0]])

zoomed_no_int_bgr = cv2.merge([zoomed_no_int_b, zoomed_no_int_g, zoomed_no_int_r])
zoomed_no_int_bgr = np.array(zoomed_no_int_bgr, dtype=np.uint8)

cv2.imshow("roi_zoomed", zoomed_no_int_bgr)
cv2.waitKey(0)
