import numpy as np
import matplotlib.pyplot as plt


# Convenience function to calculate the RMSE
def rmse(target, prediction):
    return np.sqrt(((target - prediction) ** 2).mean())


# Useful to understand the target value
def visualise_target_value(target):
    mean = np.mean(target)
    # median = np.median(diabetes.target)
    print(mean, np.std(target), np.min(target), np.max(target))
    plt.hist(target, bins=40)
    plt.xlabel("Response of interest", fontsize='x-large')
    plt.axvline(x=mean, color='black')
    plt.annotate('mean', xy=(mean+5, 25))
    plt.show()


# Convenience function to print nicely formatted training and testing error
def print_results(train_rmse, test_rmse):
    print("Train rmse: {:10.02f}\nTest rmse: {:10.02f}".format(train_rmse, test_rmse))
