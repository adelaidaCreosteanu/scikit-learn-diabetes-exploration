import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

# load dataset
diabetes = load_diabetes()
size = int(diabetes.data.shape[0])

def rmse(target, prediction):
    return np.sqrt(((target - prediction) ** 2).mean())

def rnd_permutation(train_percent, data=diabetes.data, target=diabetes.target):
    # Random indices for training and testing data
    rnd_indices = np.random.permutation(size)
    train_size = int(size*train_percent)
    test_start = int(size*0.8)  # always 20%
    train_indices = rnd_indices[:train_size]
    test_indices = rnd_indices[test_start:]
    
    train_data = data[train_indices, :]
    train_target = target[train_indices]
    
    test_data = data[test_indices, :]
    test_target = target[test_indices]
    
    return train_data, train_target, test_data, test_target


# print(np.mean(diabetes.target), np.std(diabetes.target))
# plt.figure()
# plt.hist(diabetes.target, bins=25)
# plt.show()