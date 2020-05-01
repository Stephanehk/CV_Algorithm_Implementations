import numpy as np

matrix1 = [[1,2,3], [4,5,6], [7,8,9]]

new = [[None for i in range (len(matrix1))] for j in range (len(matrix1[0]))]
print (new)
for i in range (len(matrix1)):
    for j in range (len(matrix1[0])):
        new[i][j] = i+j
        print (new[i][j])
print (new)
