import cv2
import numpy as np

def get_corner_cords(img_gray):
    img_gray = np.float32(img_gray)
    #harris corner detection
    dst = cv2.cornerHarris(img_gray,3,3,0.04)
    #threshold image to only get corenrs above threshold 0.01 time local maximum
    ret, dst = cv2.threshold(dst,0.01*dst.max(),255,0)
    dst = np.uint8(dst)
    ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
    corners = cv2.cornerSubPix(img_gray,np.float32(centroids),(5,5),(-1,-1),criteria)
    # for i in range(1, len(corners)):
    #     print(corners[i])
    return corners

def format_arr(arr):
    res = []
    for i in range (len(arr)):
        for j in range(len(arr[0])):
            res.append(arr[i][j])
    return res

def get_xy_gradient_img(img):
    kernely = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    kernelx = np.array([[1,0,-1],[1,0,-1],[1,0,-1]])
    edges_x = cv2.filter2D(img,cv2.CV_8U,kernelx)
    edges_y = cv2.filter2D(img,cv2.CV_8U,kernely)
    edges_x_y = np.dstack((edges_x,edges_y))
    return edges_x_y

def least_squares(S,T):
    try:
        return np.dot(np.linalg.inv(np.dot(S.T,S)),np.dot(S.T,T))
    except Exception as e:
        print (e)
        return [0,0]

img1 = cv2.imread("optical_flow_test_img_1.png")
img2 = cv2.imread("optical_flow_test_img_2.png")

def optical_flow(img1,img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    #get corners in image
    img1_corners = get_corner_cords(img1_gray)
    #get image gradients
    img1_xy_gradients = get_xy_gradient_img(img1_gray)


    #----------------MUST BE DONE FOR EACH DETECTED CORNER------------------------------------------------
    for corner in img1_corners:
        corner_x = int(corner[0])
        corner_y = int(corner[1])

        roi_og = img1_gray[corner_y-2: corner_y+1, corner_x-2:corner_x+1]
        roi_next_img = img2_gray[corner_y-2: corner_y+1, corner_x-2:corner_x+1]
        roi_xy_grads = img1_xy_gradients[corner_y-2: corner_y+1, corner_x-2:corner_x+1]

        #T IS NOT IN THE RIGHT FORMAT
        S = np.array(format_arr(roi_xy_grads))
        T = np.array(format_arr(roi_og - roi_next_img))

        delta_p = least_squares(S,T)
        try:
            cv2.line(img2,(corner_x,corner_y), (corner_x+int(delta_p[0]),corner_y+int(delta_p[1])), (0,255,0),2)
        except OverflowError:
            continue
        #print (delta_p)
    return img2


cap = cv2.VideoCapture(0)
prev_frame = []
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if len(prev_frame) != 0:
        cv2.imshow("frame",optical_flow(frame,prev_frame))
    else:
        cv2.imshow('frame',frame)
    prev_frame = frame
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
