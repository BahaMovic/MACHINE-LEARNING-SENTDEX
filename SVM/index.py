import numpy as np
import matplotlib.pyplot as plt


class Support_Vector_Machine:

    def fit(self,data):
        self.data = data
        dct_data = {}
        all_data = []

        for features in data:
            for feature in data[features]:
                for item in feature:
                    all_data.append(item)

        max_feature_value = max(all_data)
        min_feature_value = min(all_data)
        all_data = None

        transforms = [[1,1],
                      [-1,1],
                      [-1,-1],
                      [1,-1]]
        step_size = [max_feature_value*.01,max_feature_value*.001,max_feature_value*.0001,max_feature_value*.00001]

        for step in step_size:
            w = np.array([max_feature_value,max_feature_value])
            optimized = False
            while not optimized:
                for b in np.arange(-1*max_feature_value,max_feature_value,step):
                    for trans in transforms:
                        w_t = w*trans
                        for i in self.data:
                            optimum_option = True
                            for item in self.data[i]:
                                yi = i
                                # yi * (w_t.item)+b >=1
                                result = yi* (np.dot(w_t,item)+b)
                                if not result >= 1:
                                    optimum_option = False
                                    break
                            if optimum_option:
                                dct_data[np.sqrt(w_t[0]**2+w_t[1]**2)] = [w_t,b]
                                # print(dct_data[np.sqrt(w_t[0] ** 2 + w_t[1] ** 2)])

                    if w[0] < 0:
                        optimized = True

                    else:
                        w = w - step
              
                norms = sorted([n for n in dct_data])
                print(norms[0])
                self.w = norms[int(len(norms)/12)]
                self.b = norms[int(len(norms)/12)]
                return self.w , self.b




    def predict(self,data):
        classification = np.sign(np.dot(data,self.w)+self.b)
        return classification

data = {-1:np.array([[1,7],[2,8],[3,9],]),1:np.array([[5,1],[6,-1],[7,3]])}

# for yi in data:
#     for i in data[yi]:
#         if yi == -1:
#             plt.scatter(i[0],i[1],color="red")
#         else:
#             plt.scatter(i[0],i[1],color="blue")


df = Support_Vector_Machine()
w , b = df.fit(data)
print(w,b)

for i in data:
    for ix in data[i]:
        if i == -1:
            plt.scatter(ix[0],ix[1],color="red")
        else:
            plt.scatter(ix[0],ix[1],color="blue")

plt.plot(range(10),np.around(w*np.arange(0,10,1)+b,decimals=1))
plt.show()

