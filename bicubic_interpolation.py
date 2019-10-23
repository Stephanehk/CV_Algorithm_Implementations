import numpy as np

def cubic_kernel(val):
    #create bicubic kernel
    a = 0.5
    if val <= 1:
        val = ((a+2)*np.power(np.abs(val),3)) - (a+3)*np.power(np.abs(val),2) + 1
    if 1 < val < 2:
        val = (np.power(np.abs(val),3)) - (5*a)*np.power(np.abs(val),2) + (8*a*np.abs(val)) - (4*a)
    if val >= 2:
        val = 0
    return val

def bicubic_int (x,y,points):
    kernel = np.zeros((2,2))
    counter = 0
    for i in range (2):
        for j in range (2):
            point = points[counter]
            counter+=1
            kernel[i][j] = cubic_kernel(point)
    return kernel

#                          (0,0), (1,0), (0,1), (1,1)
print(bicubic_int (0.5,0.5,[1.3,0.6,1.2,0.2]))
