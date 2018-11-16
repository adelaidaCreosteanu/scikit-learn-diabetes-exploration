import utility
from sklearn.datasets import load_diabetes
from sklearn.decomposition import KernelPCA
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Perform PCA
transformer = KernelPCA(n_components=4, kernel='rbf', gamma=0.02)
X_transformed = transformer.fit_transform(load_diabetes().data)

# Uniformly randomly split data 80% training and 20% testing
train_data, train_target, test_data, test_target = utility.rnd_permutation(0.8, X_transformed)

# Create figure where all three models will be plotted
plt.figure(1)
plt.scatter(train_data[:,0], train_target, color='black', marker=',')
plt.figure(2)
plt.scatter(train_data[:,1], train_target, color='black', marker=',')
plt.figure(3)
plt.scatter(train_data[:,2], train_target, color='black', marker=',')
plt.figure(4)
plt.scatter(train_data[:,3], train_target, color='black', marker=',')
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
    train_rmse = utility.rmse(train_target, train_predict)

    test_predict = polynomial_regression.predict(test_data)
    test_rmse = utility.rmse(test_target, test_predict)

    print("Degree ", degree, " rmse: ", "{:10.02f}".format(train_rmse), "{:10.02f}".format(test_rmse))

    plt.figure(1)
    plt.scatter(test_data[:,0], test_predict, color=colours[count], label="d %d" % degree)
    plt.figure(2)
    plt.scatter(test_data[:,1], test_predict, color=colours[count], label="d %d" % degree)
    plt.figure(3)
    plt.scatter(test_data[:,2], test_predict, color=colours[count], label="d %d" % degree)
    plt.figure(4)
    plt.scatter(test_data[:,3], test_predict, color=colours[count], label="d %d" % degree)

# fig_1 = plt.legend(loc='lower left')
# fig_2 = plt.legend(loc='lower left')
# fig_3 = plt.legend(loc='lower left')
# fig_4 = plt.legend(loc='lower left')
plt.show()