import numpy as np

A = np.array([[1, 2, 3, 4, 5, 6]]).T
B = np.array([3, 6, 7, 8, 9, 10])

C = A - B.reshape(6, 1)

print(C)
print(B)
