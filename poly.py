import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# load dataset
diabetes = load_diabetes()

size = int(diabetes.data.shape[0])
rnd_indices = np.random.permutation(size)
train_size = int(size*0.8)
train_indices = rnd_indices[:train_size]
test_indices = rnd_indices[train_size:]

train_data = preprocessing.scale(diabetes.data[train_indices, :])
test_data = preprocessing.scale(diabetes.data[test_indices, :])
train_target = diabetes.target[train_indices]
test_target = diabetes.target[test_indices]

# Create figure where all three models will be plotted
plt.figure()

colours = ['teal', 'yellowgreen', 'gold']

for count, degree in enumerate([2,5,10]):
    # Polynomially expand every feature of the data:
    polynomial_regression = Pipeline((
        ("poly_features", PolynomialFeatures(degree=degree, include_bias=True)),
        ("sgd_reg", LinearRegression()),
    ))
    polynomial_regression.fit(train_data, train_target)

    # Print RMSE on training and testing data
    train_predict = polynomial_regression.predict(train_data)
    train_rmse = np.sqrt(((train_target - train_predict) ** 2).mean())

    test_predict = polynomial_regression.predict(test_data)
    test_rmse = np.sqrt(((test_target - test_predict) ** 2).mean())

    print("Degree ", degree, " rmse: ", "{:10.02f}".format(train_rmse), "{:10.02f}".format(test_rmse))

    plt.plot(test_data, test_predict, color=colours[count], label="d %d" % degree)
    
plt.legend(loc='lower left')
plt.show()