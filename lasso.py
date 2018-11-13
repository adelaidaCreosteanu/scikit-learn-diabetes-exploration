import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline

# load dataset
diabetes = load_diabetes()

size = int(diabetes.data.shape[0])
rnd_indices = np.random.permutation(size)
train_size = int(size*0.8)
train_indices = rnd_indices[:train_size]
test_indices = rnd_indices[train_size:]

train_data = diabetes.data[train_indices, :]
test_data = diabetes.data[test_indices, :]
train_target = diabetes.target[train_indices]
test_target = diabetes.target[test_indices]

clf = Lasso(alpha=0.1)
clf.fit(train_data)