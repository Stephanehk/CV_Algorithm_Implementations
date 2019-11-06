import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

import map_resized_cords

img = cv2.imread("/Users/2020shatgiskessell/Downloads/test_img.png")
img = cv2.resize(img, (50,50))

def match_coordinates_coefs(img, target_size):
    #TODO: IDK HOW MUCH OF THIS HOLDS TRUE FOR OTHER IMAGE DIMENSIONS
    t_h, t_w = target_size
    h,w, _ = img.shape
    #define original image ending coordinates
    h = h - 0.5
    w = w - 0.5
    #target images top left x,y coordinates
    t_sx, t_sy = -0.5, -0.5
    #target images bottom right x,y coordinates
    t_ex, t_ey = t_h-0.5, t_w-0.5
    #following systems of equations
    #ax + b = Y
    #-0.5*a + b = -0.5
    #t_ex*a + b = h

    #solving for a and b:
    a = (-0.5 - h)/(-0.5 - t_ex)
    b = -0.5 - (a*-0.5)
    return a,b

def old_cords2new_cords(x,a,b):
    #ax + b = Y
    return (a*x) + b

def bilinear_interpolate (img, target_size):
    og_img = img
    # #TODO: ADD IMAGE PADDING
    # a_old_pixel_cords = []
    # a_new_pixel_cords = []
    #
    h,w,_ = img.shape
    #
    # #TODO SOMETHING IS MISSING IN TERMS OF TRANSLATING OLD CORDS TO NEW CORDS
    # #PLOT DOES NOT MAKE SENCE!!!!
    #
    # #get coordinate matching coefficients
    # a,b = match_coordinates_coefs(img, target_size)
    # #iterate over every pixel in img
    # for i in range (h):
    #     for j in range (w):
    #         old_img_cords = [i,j]
    #         new_img_cords = [np.round(old_cords2new_cords(i,a,b),2),np.round(old_cords2new_cords(j,a,b),2)]
    #         a_old_pixel_cords.append(old_img_cords)
    #         a_new_pixel_cords.append(new_img_cords)

    dif_h, dif_w = target_size
    img, a_new_pixel_cords, a_old_pixel_cords = map_resized_cords.get_uninterpolated_scaled_img(img, dif_h/h)
    #--------------------------PLOT-------------
    fig=plt.figure(figsize=(10,10))
    plt.scatter([x[0] for x in a_old_pixel_cords], [x[1] for x in a_old_pixel_cords],color="r")
    plt.scatter([x[0] for x in a_new_pixel_cords], [x[1] for x in a_new_pixel_cords], color = "b")
    plt.savefig("new_vs_old_cords.png")

    new_points = {}
    #build KDTree of old pixel cords
    tree = KDTree(a_old_pixel_cords)
    #interpolate all new pixels
    for point in a_new_pixel_cords:
        #Find closest point to each new coordinate point
        dist, ind = tree.query(point, k = 4)
        closest_pixels = [np.array(a_old_pixel_cords[i]) for i in ind]
        #sort closest points by x cordinate
        closest_pixels.sort(key=lambda x:x[0])

        #interplate q1 and q2

        #same x cord as old point, same y cord as new point
        q1_cords = np.array([closest_pixels[0][0], point[1]])
        q2_cords = np.array([closest_pixels[2][0], point[1]])

        #get distances to figure out how much of r,g,b should be in q1 and q2
        #SOMETHING WRONG WITH DISTANCES BEING USED
        q1_d_1 = (np.linalg.norm(q1_cords - closest_pixels[1]))/2
        q1_d_2 = (np.linalg.norm(q1_cords - closest_pixels[0]))/2
        q2_d_1 = (np.linalg.norm(q2_cords - closest_pixels[3]))/2
        q2_d_2 = (np.linalg.norm(q2_cords - closest_pixels[2]))/2

        q1 = [q1_d_1*img[closest_pixels[0][0], closest_pixels[0][1]][0] + q1_d_2*img[closest_pixels[1][0], closest_pixels[1][1]][0], q1_d_1*img[closest_pixels[0][0], closest_pixels[0][1]][1] + q1_d_2*img[closest_pixels[1][0], closest_pixels[1][1]][1], q1_d_1*img[closest_pixels[0][0], closest_pixels[0][1]][2] + q1_d_2*img[closest_pixels[0][0], closest_pixels[1][1]][2]]
        q2 = [q2_d_1*img[closest_pixels[2][0], closest_pixels[2][1]][0] + q2_d_2*img[closest_pixels[3][0], closest_pixels[3][1]][0], q2_d_1*img[closest_pixels[2][0], closest_pixels[2][1]][1] + q2_d_2*img[closest_pixels[3][0], closest_pixels[3][1]][1], q2_d_1*img[closest_pixels[2][0], closest_pixels[2][1]][2] + q2_d_2*img[closest_pixels[3][0], closest_pixels[3][1]][2]]
        #interpolate point
        #get distance from q1 and q2 to new point
        q1_point_d = (np.linalg.norm(point - q2_cords))/2
        q2_point_d = (np.linalg.norm(point - q1_cords))/2

        new_point_val = (int(q1_point_d * q1[0] + q2_point_d*q2[0]), int(q1_point_d * q1[1] + q2_point_d*q2[1]), int(q1_point_d * q1[2] + q2_point_d*q2[2]))
        new_points[tuple(point)] = new_point_val
        if point == [1,3]:
            print (new_point_val)
    #add old pixels to dictionary
    for point in a_old_pixel_cords:
        x,y = point
        new_points[tuple(point)] = img[x][y]

    #reconstruct image from dictionary
    resized_image = np.zeros((dif_h, dif_w,3),np.uint8)
    coordinates = list(new_points.keys())
    coordinates = sorted(coordinates, key = lambda tup: tup[1])
    coordinates = sorted(coordinates, key = lambda tup: tup[0])
    for cords in coordinates:
        x,y = cords
        resized_image[x][y] = list(new_points.get(cords))
    cv2.imwrite("resized_image.png", resized_image)
    cv2.imwrite("no_int_resized_image.png", img)
    #cv2.imshow("og_img", og_img)
    #cv2.waitKey(0)

bilinear_interpolate (img, (100,100))
