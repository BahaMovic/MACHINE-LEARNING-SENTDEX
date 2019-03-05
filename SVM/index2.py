import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

def SVM(data):
    all_data = []
    for i in data[-1]:
       all_data.append(i[1])
    max_n_p = max(all_data)
    all_data = []
    for i in data[1]:
        all_data.append(i[1])
    min_p_p = min(all_data)

    x1_x2 = min_p_p - max_n_p
    # print(data[-1][:,0])
    steps = [1]
    w = ((data[-1][:, 0] * data[-1][:, 1]).mean() - data[-1][:, 0].mean() * data[-1][:, 1].mean()) / (
                (data[-1][:, 0] ** 2).mean() - (data[-1][:, 0].mean()) ** 2)
    P = (data[-1][:, 1]).mean() - w * mean(data[-1][:, 0])

    w2 = ((data[1][:, 0] * data[1][:, 1]).mean() - data[1][:, 0].mean() * data[1][:, 1].mean()) / (
                (data[1][:, 0] ** 2).mean() - (data[1][:, 0].mean()) ** 2)
    P2 = (data[1][:, 1]).mean() - w2 * mean(data[1][:, 0])

    w = [w,w]
    w2 = [w2,w2]
    for step in steps:
        complitied = False
        while not complitied:
            for group in data:
                for row in data[group]:
                    group = group * 1.0
                    mult =group*(np.dot(w,row)+P)
                    if mult < 1:
                        w = np.array(w) - step
                        P = P - step
                        break
            complitied = True
    print(w,P,x1_x2)
    return w , P ,np.sqrt(x1_x2**2)/2






data = {-1:np.array([[1,7],[2,8],[3,10],]),1:np.array([[5,3],[6,-1],[7,2]])}
w, b , p= SVM(data)

for i in data:
    for ix in data[i]:
        if i == -1:
            plt.scatter(ix[0],ix[1],color="red")
        else:
            plt.scatter(ix[0],ix[1],color="blue")
plt.plot(range(10),np.around(w[0]*np.arange(0,10,1)+b,decimals=1))

plt.show()