import numpy as np
import cv2

img = cv2.imread("/Users/2020shatgiskessell/Downloads/test_img.png")
roi = cv2.resize(img,(50,50))

# img = cv2.imread("/Users/2020shatgiskessell/Downloads/test_img.png")
#roi = img[200:210, 800:810]
#img[200:600, 800:1200]
#img[200:210, 800:810]

def basic_int (matrix):
    for i in range (len(matrix)):
        for j in range(len(matrix[0])):
            if np.sum(matrix[i][j]) == 0:
                matrix[i][j] = prev
            else:
                prev = matrix[i][j]
    return matrix

def get_uninterpolated_scaled_img(roi, scale_factor):
    h,w = scale_factor
    #isolate each image channel
    b, g, r = cv2.split(roi)
    b = roi[...,0]
    g = roi[...,1]
    r = roi[...,2]
    #create Kronecker product kernel
    kron_kernel = np.zeros((int(h), int(w)))
    kron_kernel[0][0] = 1
    #calculate Kronecker product for each channel
    zoomed_no_int_b = np.kron(b, kron_kernel)
    zoomed_no_int_g = np.kron(g, kron_kernel)
    zoomed_no_int_r = np.kron(r, kron_kernel)
    #add channels back together
    zoomed_no_int_bgr = cv2.merge([zoomed_no_int_b, zoomed_no_int_g, zoomed_no_int_r])
    zoomed_no_int_bgr = np.array(zoomed_no_int_bgr, dtype=np.uint8)

    #get all uninterpolated and old pixel cords
    unint_pixels = np.where(zoomed_no_int_bgr == [0,0,0])
    old_pixels = np.where(zoomed_no_int_bgr != 0)

    #get everything in correct format
    print ("1")
    unint_pixel_cords = set(zip(unint_pixels[0], unint_pixels[1]))
    unint_pixel_cords = [list(pixel) for pixel in unint_pixel_cords]
    old_pixel_cords = set(zip(old_pixels[0], old_pixels[1]))
    old_pixel_cords = [list(pixel) for pixel in old_pixel_cords]
    print ("2")
    return zoomed_no_int_bgr, unint_pixel_cords, old_pixel_cords

#zoomed_no_int_bgr, _, _ = get_uninterpolated_scaled_img(roi, (1, 1))
# cv2.imshow("roi_zoomed", zoomed_no_int_bgr)
# cv2.imshow("roi", roi)
# cv2.waitKey(0)
