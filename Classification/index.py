import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from collections import Counter
# from sklearn import neighbors,model_selection
#
# dataSet = pd.read_csv('breast-cancer-wisconsin.data.txt')
# dataSet.columns = ['id','clump','unif_cell_size','unif_cell_shape','merg','single','bare','bland','normal','mitoses','class']
# dataSet.drop('id',1,inplace=True)
# dataSet.replace('?',-999999,inplace=True)
# X = np.array(dataSet.drop('class',axis=1,inplace=False))
# y = np.array(dataSet['class'])
#
# X_Train , X_Test , Y_Train , Y_Test = model_selection.train_test_split(X,y,test_size=0.2)
#
# clf = neighbors.KNeighborsClassifier()
# clf.fit(X_Train,Y_Train)
# accuracy = clf.score(X_Test,Y_Test)
# print(accuracy)


dataSet = {'k':[[1,2,4],[2,1,3],[3,2,2]],'r':[[5,5,7],[6,7,6],[7,5,8],[3,4,1]]}

def get_fit_KNearestNeighbor(data,predict,k=3):
    if len(data) >= k:
        return "warning data length shouldn't be more than k"
    distances = []
    for group in data:
        for i in data[group]:
            count = 0
            y_ind = 0
            for ii in i:
                count += (ii - predict[y_ind])**2
                y_ind += 1
            distance = math.sqrt(count)
            distances.append([distance,group])
    vote = [i[1] for i in sorted(distances)[:k]]
    vote = Counter(vote).most_common(1)[0][0]
    return vote

votes = 0
count = 0

for group in dataSet:
    for i in dataSet[group]:
        if get_fit_KNearestNeighbor(dataSet,i,3) == group:
            votes += 1
        count +=1
print('Accuracy : ',votes/count)

print(get_fit_KNearestNeighbor(dataSet,[8,7,6],k=3))



# plt.scatter(4,2,color="black",s=200)
# plt.scatter([i[0] for i in dataSet['k']],[i[1] for i in dataSet['k']],color="red")
# plt.scatter([i[0] for i in dataSet['r']],[i[1] for i in dataSet['r']], color="blue")
plt.show()