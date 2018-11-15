import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV

def rmse(target, prediction):
    return np.sqrt(((target - prediction) ** 2).mean())

# load dataset
diabetes = load_diabetes()

size = int(diabetes.data.shape[0])

sumAlpha = 0

for i in range(50):
    rnd_indices = np.random.permutation(size)
    train_size = int(size*0.8)
    train_indices = rnd_indices[:train_size]
    test_indices = rnd_indices[train_size:]

    train_data = diabetes.data[train_indices, :]
    test_data = diabetes.data[test_indices, :]
    train_target = diabetes.target[train_indices]
    test_target = diabetes.target[test_indices]

    # alpha=0 is equivalent to LinearRegression
    # it defaults to 1
    # for alpha in [1e-07, 0.01, 0.05, 0.1, 0.2, 0.5, 1]:
    #     train_rmse_sum = 0
    #     test_rmse_sum = 0
        
    #     for i in range(20):
    #         clf = linear_model.Lasso(alpha=alpha)
    #         clf.fit(train_data, train_target)

    #         # Print RMSE on training and testing data
    #         train_predict = clf.predict(train_data)
    #         train_rmse_sum += rmse(train_target, train_predict)

    #         test_predict = clf.predict(test_data)
    #         test_rmse_sum += rmse(test_target, test_predict)

    #     print("Alpha: ", alpha, "\nRMSE: ", "{:10.02f}".format(train_rmse_sum/20), "{:10.02f}".format(test_rmse_sum/20))

    # X = diabetes.data
    # y = diabetes.target

    param_grid = [
        {'alpha': [1e-04, 1e-03, 0.005, 0.01, 0.02, 0.04, 0.05, 0.06, 0.07, 0.8, 0.8, 0.1, 0.15, 0.2]}
    ]

    lasso = linear_model.Lasso(max_iter=1000000000, tol=0.0000001)
    # cross-validation with 5 folds
    grid_search = GridSearchCV(lasso, param_grid, scoring=['neg_mean_squared_error','neg_mean_absolute_error', 'neg_mean_squared_log_error'],
    cv=10, refit='neg_mean_squared_error')
    grid_search.fit(train_data, train_target)
    sumAlpha += grid_search.best_params_['alpha']
    print("Score ", np.sqrt(-grid_search.best_score_))

optimalAlpha = sumAlpha/50
print("\nBest alpha: ", optimalAlpha)
# Best alpha is between 0.001 and 0.1. 0.038 after 50 iterations of grid_search

las = linear_model.Lasso(alpha=optimalAlpha)
las.fit(train_data, train_target)
print(las.coef_)
