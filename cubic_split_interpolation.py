import numpy as np
import cv2

img = cv2.imread("/Users/2020shatgiskessell/Downloads/test_img.png")
roi = img[200:600, 800:1200]
#img[200:600, 800:1200]
#img[200:210, 800:810]
#img[200:600, 800:1200]

def get_uninterpolated_scaled_img(roi, scale_factor):
    #isolate each image channel
    b, g, r = cv2.split(roi)
    b = roi[...,0]
    g = roi[...,1]
    r = roi[...,2]
    #create Kronecker product kernel
    kron_kernel = np.zeros((scale_factor, scale_factor))
    kron_kernel[0][0] = 1
    #calculate Kronecker product for each channel
    zoomed_no_int_b = np.kron(b, kron_kernel)
    zoomed_no_int_g = np.kron(g, kron_kernel)
    zoomed_no_int_r = np.kron(r, kron_kernel)
    #add channels back together
    zoomed_no_int_bgr = cv2.merge([zoomed_no_int_b, zoomed_no_int_g, zoomed_no_int_r])
    zoomed_no_int_bgr = np.array(zoomed_no_int_bgr, dtype=np.uint8)
    return zoomed_no_int_bgr

zoomed_no_int_bgr = get_uninterpolated_scaled_img(roi, 3)
cv2.imshow("roi_zoomed", zoomed_no_int_bgr)
cv2.waitKey(0)
