import numpy as np

data = np.load('dataset_expand.npy')
X = np.array(data)
mean_flow = np.mean(X[:, :, 0])
std_flow = np.std(X[:, :, 0])


print(data)
print(type(data))
print(data.shape)
print(mean_flow)
print(std_flow)
