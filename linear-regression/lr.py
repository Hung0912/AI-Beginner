import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Random data
A = np.array([[2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46]]).T
y = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T

ones = np.ones_like(A)
A1 = np.concatenate((A, ones), axis=1)

x = np.linalg.inv(((A1.transpose()).dot(A1))).dot(A1.transpose()).dot(y)

X = np.arange(0, 46)
Y = X * x[0][0] + x[1][0]

plt.plot(X, Y)
plt.plot(A, y, 'ro')
plt.show()
