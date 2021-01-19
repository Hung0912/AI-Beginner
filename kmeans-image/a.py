import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

img = plt .imread('a.jpg')
# print(img.shape)
height = img.shape[0]
width = img.shape[1]

img = img.reshape(height*width, 3)
# print(img.shape)

# Kmeans img
kmeans = KMeans(n_clusters=4).fit(img)
labels = kmeans.labels_
# labels = kmeans.predict(img)
clusters = kmeans.cluster_centers_

img2 = np.zeros_like(img)

for i in range(len(img2)):
    img2[i] = clusters[labels[i]]

img2 = img2.reshape(height, width, 3)
print(img2)
plt.imshow(img2)
plt.show()
