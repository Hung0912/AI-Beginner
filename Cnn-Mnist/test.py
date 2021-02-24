import numpy as np

a = np.array([[1, 2, 3]])  # 1x3
b = np.array([2, 3])  # 3,
b = b[np.newaxis]
c = a.T @ b
d = np.dot(a.T, b)
print(c)
print(d)
# permutation = np.random.permutation(1000)

# print(permutation)
