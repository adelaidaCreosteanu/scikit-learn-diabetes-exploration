import utility as u
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


# Plots the RMSE against increasing the alpha value.
# dataset has to be a Bunch with data and target attributes
def plot_alphas_lasso(dataset):
    x_train, x_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
    train_errors, test_errors = [], []
    alphas = np.linspace(1e-06,3,1000)

    for _ in alphas:
        lasso = linear_model.Lasso(alpha=m)
        lasso.fit(x_train, y_train)

        train_rmse = u.rmse(y_train, lasso.predict(x_train))
        test_rmse = u.rmse(y_test, lasso.predict(x_test))
        train_errors.append(train_rmse)
        test_errors.append(test_rmse)

    plt.plot(alphas, train_errors, 'r-', linewidth=2, label='training')
    plt.plot(alphas, test_errors, 'b-', linewidth=3, label='validation')
    plt.xlabel('alpha')
    plt.ylabel('RMSE')
    plt.legend(loc='lower right')
    plt.show()


# Finds optimal alpha using grid search and cross-validation on the given dataset
def optimal_alpha(dataset):
    param_grid = [
        {'alpha': np.logspace(-6,0.5,1000)}
    ]

    lasso = linear_model.Lasso()
    # cross-validation with 5 folds
    grid_search = GridSearchCV(lasso, param_grid, cv=5)
    grid_search.fit(dataset.data, dataset.target)

    return grid_search.best_params_['alpha']
