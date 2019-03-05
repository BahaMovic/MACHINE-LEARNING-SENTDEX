import numpy as np
import matplotlib.pyplot as plt
class KMeans:

    def __init__(self,k=2,max_iter=300):
        self.k = k
        self.max_iter = max_iter

    def fit(self,data):
        self.data= data
        self.centroids = []
        for i in range(self.k):
            self.centroids.append(data[i])

        for i in range(self.max_iter):
            self.classifications = {}
            for ki in range(self.k):
                self.classifications[ki] = []

            for d in data:
                distance= [np.linalg.norm(d- centroid) for centroid in self.centroids]
                self.classifications[distance.index(min(distance))].append(d)
            for cls in range(len(self.centroids)):
                self.centroids[cls] = np.average(self.classifications[cls], axis=0)
        print(self.classifications)
        prev_centroids = dict(self.centroids)
        print(prev_centroids)

        for i in data:
            plt.scatter(i[0], i[1], color="red")
        plt.scatter(self.centroids[0][0], self.centroids[0][1], color="blue", s=200)
        plt.scatter(self.centroids[1][0], self.centroids[1][1], color="blue", s=200)
        plt.show()



m = KMeans(k=2,max_iter=300)
data = np.array([[1,2],[2,1],[3,3],[4,2],[5,29],[6,18],[7,25],[8,23]])
m.fit(data)
