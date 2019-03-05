import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
data = np.array([[1,2],[2,1],[3,3],[4,2],[5,29],[6,18],[7,25],[8,23]])

cls = MeanShift()
cls.fit(data)

centroids = cls.cluster_centers_
for i in data:
    plt.scatter(i[0], i[1], color="red")

plt.scatter(centroids[0][0], centroids[0][1], color="blue", s=200)
plt.scatter(centroids[1][0], centroids[1][1], color="blue", s=200)
plt.show()

