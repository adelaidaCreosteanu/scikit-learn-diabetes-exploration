import utility
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def run_linear_regression(diabetes, iterations):
    train_rmse_sum = 0
    test_rmse_sum = 0

    for _ in range(iterations):
        # Uniformly randomly split data 80% training and 20% testing
        train_data, test_data, train_target, test_target = train_test_split(diabetes.data, diabetes.target, test_size=0.2)
        
        lin_reg = LinearRegression()
        lin_reg.fit(train_data, train_target)

        # Calculate RMSE on training and testing data
        train_predict = lin_reg.predict(train_data)
        train_rmse_sum += utility.rmse(train_target, train_predict)

        test_predict = lin_reg.predict(test_data)
        test_rmse_sum += utility.rmse(test_target, test_predict)

    # Print results
    avg_train_rmse = "{:10.02f}".format(train_rmse_sum/iterations)
    avg_test_rmse = "{:10.02f}".format(test_rmse_sum/iterations)
    print("Train rmse: {0}\nTest rmse: {1}".format(avg_train_rmse, avg_test_rmse))
