import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

# Random data
A = np.array([[2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35, 37, 40, 46]]).T
y = np.array([[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]]).T

lr = linear_model.LinearRegression()
lr.fit(A, y)

coeffiecient = lr.coef_  # he so a
intercept = lr.intercept_  # b

X = np.array([[1, 46]]).T
Y = X * coeffiecient + intercept

print(lr.score(A, y))
plt.plot(X, Y)
plt.plot(A, y, 'ro')
plt.show()
