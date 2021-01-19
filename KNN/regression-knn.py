import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors

X = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
              15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]).reshape(-1, 1)
y = [2, 5, 7, 9, 11, 16, 19, 23, 22, 29, 29, 35,
     37, 40, 46, 42, 39, 31, 30, 28, 20, 15, 10, 6]
print(X)
plt.plot(X, y, 'ro')

x0 = np.linspace(3, 25, 10000).reshape(-1, 1)
y0 = []

knn = neighbors.KNeighborsRegressor(n_neighbors=3)
knn = knn.fit(X, y)
y0 = knn.predict(x0)

plt.plot(x0, y0, color='blue')
plt.show()
