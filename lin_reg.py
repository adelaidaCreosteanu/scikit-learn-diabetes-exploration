import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# Open Anaconda prompt and run conda activate Diabetes
# cd into this folder (C:\Users\User\OneDrive\Docs Uni\Y3S1\ML technologies\Cw1)
# and run main.py

# load dataset
diabetes = load_diabetes()
size = diabetes.data.shape[0]

train_rmse_sum = 0
test_rmse_sum = 0

# Uniformly randomly split data 80% training and 20% testing
for i in range(20):
    rnd_indices = np.random.permutation(size)
    train_size = int(size * 0.8)
    train_indices = rnd_indices[:train_size]
    test_size = int(size * 0.8)
    test_indices = rnd_indices[test_size:]

    train_data = preprocessing.scale(diabetes.data[train_indices, :])
    test_data = preprocessing.scale(diabetes.data[test_indices, :])
    train_target = diabetes.target[train_indices]
    test_target = diabetes.target[test_indices]

    # Linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(train_data, train_target)

    # Print RMSE on training and testing data
    train_predict = lin_reg.predict(train_data)
    train_rmse_sum += np.sqrt(((train_target - train_predict) ** 2).mean())

    test_predict = lin_reg.predict(test_data)
    test_rmse_sum += np.sqrt(((test_target - test_predict) ** 2).mean())

print("rmse: ", "{:10.02f}".format(train_rmse_sum/20), "{:10.02f}".format(test_rmse_sum/20))

# plt.figure()
# plt.plot(train_data[:,0], train_target, "bo")
# plt.plot(train_data[:,1], train_target, "ro")
# plt.plot(train_data[:,2], train_target, "go")
# plt.figure()
# plt.plot(train_data[:,3], train_target, "yo")
# plt.figure()
# plt.plot(train_data[:,4], train_target, "b*")
# plt.plot(train_data[:,5], train_target, "r*")
# plt.plot(train_data[:,6], train_target, "g*")
# plt.figure()
# plt.plot(train_data[:,7], train_target, "y*")
# plt.plot(train_data[:,8], train_target, "b.")
# plt.plot(train_data[:,9], train_target, "r.")
# plt.show()

# Print model parameters
# print (' '.join("{:10.02f}".format(x) for x in lin_reg.coef_))
# print ("{:10.02f}".format(lin_reg.intercept_))