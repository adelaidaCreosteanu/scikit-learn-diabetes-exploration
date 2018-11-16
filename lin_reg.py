import numpy as np
import utility
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Open Anaconda prompt and run conda activate Diabetes
# cd into this folder (C:\Users\User\OneDrive\Docs Uni\Y3S1\ML technologies\Cw1)
# and run main.py

train_rmse_sum = 0
test_rmse_sum = 0

for i in range(20):
    # Uniformly randomly split data 80% training and 20% testing
    train_data, train_target, test_data, test_target = utility.rnd_permutation(0.8)
    
    # Linear regression
    lin_reg = LinearRegression()
    lin_reg.fit(train_data, train_target)

    # Print RMSE on training and testing data
    train_predict = lin_reg.predict(train_data)
    train_rmse_sum += utility.rmse(train_target, train_predict)

    test_predict = lin_reg.predict(test_data)
    test_rmse_sum += utility.rmse(test_target, test_predict)

print("rmse: ", "{:10.02f}".format(train_rmse_sum/20), "{:10.02f}".format(test_rmse_sum/20))

# Print model parameters
# print (' '.join("{:10.02f}".format(x) for x in lin_reg.coef_))
# print ("{:10.02f}".format(lin_reg.intercept_))