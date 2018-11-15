import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.decomposition import KernelPCA

# load dataset
diabetes = load_diabetes()
size = int(diabetes.data.shape[0])
# Random indices for training and testing data
rnd_indices = np.random.permutation(size)
train_size = int(size*0.8)
train_indices = rnd_indices[:train_size]
test_indices = rnd_indices[train_size:]

# Perform PCA
transformer = KernelPCA(n_components=5, kernel='rbf', gamma=0.02)
X_transformed = transformer.fit_transform(diabetes.data)

train_data = X_transformed[train_indices, :]
test_data = X_transformed[test_indices, :]
train_target = diabetes.target[train_indices]
test_target = diabetes.target[test_indices]

tree = DecisionTreeRegressor(max_depth=5)
tree.fit(train_data, train_target)

# Print RMSE on training and testing data
train_predict = tree.predict(train_data)
train_rmse = np.sqrt(((train_target - train_predict) ** 2).mean())

test_predict = tree.predict(test_data)
test_rmse = np.sqrt(((test_target - test_predict) ** 2).mean())

print("Rmse: ", "{:10.02f}".format(train_rmse), "{:10.02f}".format(test_rmse))

export_graphviz(tree, max_depth=2, out_file='tree.dot', rounded=True, filled=True)