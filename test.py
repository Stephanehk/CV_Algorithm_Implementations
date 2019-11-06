import numpy as np

matrix = [[1,2,3], [4,5,6], [7,8,9]]
padded_matrix = np.pad(matrix, (1,1), "edge")
print (padded_matrix)
