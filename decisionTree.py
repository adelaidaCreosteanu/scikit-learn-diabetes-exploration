import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.decomposition import KernelPCA

# load dataset
diabetes = load_diabetes()
size = int(diabetes.data.shape[0])
train_size = int(size*0.8)

def rmse(target, prediction):
    return np.sqrt(((target - prediction) ** 2).mean())

def rnd_permutation(data, target):
    # Random indices for training and testing data
    rnd_indices = np.random.permutation(size)
    train_indices = rnd_indices[:train_size]
    test_indices = rnd_indices[train_size:]
    
    train_data = data[train_indices, :]
    train_target = target[train_indices]
    
    test_data = data[test_indices, :]
    test_target = target[test_indices]
    
    return train_data, train_target, test_data, test_target

# Perform PCA
transformer = KernelPCA(n_components=5, kernel='rbf', gamma=0.02)
X_transformed = transformer.fit_transform(diabetes.data)

for depth in range(1,6):
    best_tree = None
    train_rmse = None
    best_test_rmse = sys.float_info.max

    for iterations in range(100):
        train_data, train_target, test_data, test_target = rnd_permutation(X_transformed, diabetes.target)
        tree = DecisionTreeRegressor(max_depth=depth)
        tree.fit(train_data, train_target)

        # Evaluate tree
        test_rmse = rmse(test_target, tree.predict(test_data))
        if (test_rmse < best_test_rmse):
            best_test_rmse = test_rmse
            best_tree = tree
            train_rmse = rmse(train_target, tree.predict(train_data))
    
    # Print RMSE of best tree
    print("Tree of depth: ", depth)
    print(best_tree.max_depth)
    print("Rmse: ", "{:10.02f}".format(train_rmse), "{:10.02f}".format(best_test_rmse))

    # Save visualisation
    file_name = 'tree' + str(depth) + '.dot'
    export_graphviz(best_tree, max_depth=depth, out_file=file_name, rounded=True, filled=True)
