import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = np.array([[1,2],[2,1],[3,3],[4,2],[5,29],[6,18],[7,25],[8,23]])

# cls = KMeans(n_clusters=2)
# cls.fit(data)
# centroid = cls.cluster_centers_
# for i in data:
#     plt.scatter(i[0],i[1],color="red")
# plt.scatter(centroid[:,0],centroid[:,1],color="blue",s=200)
# plt.show()


def K_means_cluster(data,k=2,limit=5):
    ks = []
    for i in range(k):
        ks.append([np.random.randint(0,limit),np.random.randint(0,limit)])
    data_closest_cluster = []

    for row in data:
        inital_k = 0
        for k in ks:
            inital_k = np.sqrt((row[0]-k[0])**2 + (row[1]-k[1])**2)
            data_closest_cluster.append([row,k,inital_k])
    data_cluters = []
    for i in np.arange(0,len(data_closest_cluster),2):
        if data_closest_cluster[i][-1] < data_closest_cluster[i-1][-1]:
            data_cluters.append([data_closest_cluster[i][-1],data_closest_cluster[i][0],data_closest_cluster[i][1]])
        else:
            data_cluters.append([data_closest_cluster[i][-1],data_closest_cluster[i][0],data_closest_cluster[i-1][1]])
    groups = []

    for i in data_cluters:
        for k in range(len(ks)):
            if i[-1] == ks[k]:
                groups.append([k,i[-2]])
    keys_1 = [0,0]
    keys_2 = [0,0]
    print(groups)
    count1 = 0
    count2 = 0
    for i in range(len(groups)):
        if groups[i][0] == 0:
            count1 += 1
            keys_1 += groups[i][1]
        else:
            keys_2 += groups[i][1]
            # print(groups[i][1])
            count2 += 1

        ks[0] = keys_1
        ks[1] = keys_2
    print(keys_2)
    # return ks[0] , ks[1]
    return keys_1/count1 , keys_2/count2


centroid1 , centroid2 =K_means_cluster(data,2,limit=10)

for i in data:
    plt.scatter(i[0],i[1],color="red")
plt.scatter(centroid1[0],centroid1[1],color="blue",s=200)
plt.scatter(centroid2[0],centroid2[1],color="blue",s=200)
plt.show()



