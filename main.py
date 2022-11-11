
from custom_mlp import CustomMLP
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
X = data.data
y = data.target

new_y = []
new_x = []
for xi, yi in zip(X,y):
    # if yi in [0,1]:
    #     new_y.append(yi)
    #     new_x.append(xi)
    if yi == 0:
        new_y.append([1, 0])
        new_x.append(xi)
    if yi == 1:
        new_y.append([0, 1])
        new_x.append(xi)

new_y = np.array(new_y)
new_x = np.array(new_x)
mlp = CustomMLP(hidden_layer_size = (6,3),   activations = ('sigmoid','softmax'),learning_rate = 0.38, alpha=0.01, loss='categorical')

mlp.fit(X,y,verbose=1,epochs=550,batch_size=4)


x_pre_0_10 = mlp.predict(X[0:10])
x_pre_70_80 = mlp.predict(X[70:80])
# x_pre_130_140 = mlp.predict(X[130:140])
print(np.argmax(x_pre_0_10, axis=1))
print(np.argmax(x_pre_70_80, axis=1))
# print(np.argmax(x_pre_130_140, axis=1))
