import numpy as np
import pandas as pd
import h5py
import torch

class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        if type(data) == torch.Tensor and type(self.mean) == np.ndarray:
            self.std = torch.from_numpy(self.std).to(data.device).type(data.dtype)
            self.mean = torch.from_numpy(self.mean).to(data.device).type(data.dtype)
        return (data * self.std) + self.mean
#
f_bike = h5py.File('data.h5','r')
f_bike.keys()
print(f_bike)
print([key for key in f_bike.keys()])

f_graph = h5py.File('matrix.h5','r')
f_graph.keys()
print([key for key in f_graph.keys()])
bike_graph = f_graph['dis_bb'][:]
# print(f_graph['pcc_bb'][:].shape, f_graph['pcc_bb'][:])
# print(f_graph['trans_bb'][:].shape, f_graph['trans_bb'][:])
bike_drop = np.expand_dims(f_bike['bike_drop'][:], axis=-1)
bike_pick = np.expand_dims(f_bike['bike_pick'][:], axis=-1)

print(bike_drop[bike_drop==0].shape, bike_pick[bike_pick==0].shape)

bike_demand = np.concatenate([bike_drop, bike_pick], axis=-1)
np.savez('nyc_bike0.npz', data=bike_demand)

datas = np.load('nyc_bike0.npz')
print(datas.files)
print(datas['data'])
print(datas['data'].shape)

np.savetxt("nyc_bike0.csv", bike_graph, delimiter=",")
A = pd.read_csv("nyc_bike0.csv", header=None).values.astype(np.float32)
print(A.shape)
print(A)


k = datas['data']
print(k.dtype, k[k==0].shape, k[k<0.1].shape)
scaler_data = StandardScaler(k.mean(), k.std())
k = torch.FloatTensor(k)
trans1 = scaler_data.transform(k)
trans2 = scaler_data.inverse_transform(trans1)
#
print(trans2.dtype, trans2[trans2==0].shape)