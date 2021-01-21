from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

X0 = np.array([[2], [4], [5], [6], [10], [11], [12], [20], [30], [36]])
y0 = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1]]).T

lr = LogisticRegression().fit(X0, y0)
coeffiecient = lr.coef_[0][0]
intercept = lr.intercept_[0]
# print(coeffiecient, intercept)
x0_gd = np.linspace(0, 46, 1000)
y0_sklearn = 1/(1+np.exp(-(x0_gd * coeffiecient + intercept)))

plt.plot(X0, y0, 'ro')
plt.plot(x0_gd, y0_sklearn, color="green")
plt.show()
