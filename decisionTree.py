import utility as u
import numpy as np
from sys import float_info
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.model_selection import train_test_split


# Find best decision tree over niter random permutations of the data.
# A visualisation will be saved if dotfile is a file name with ".dot" extension
def best_tree(diabetes, niter=10, dotfile=None):
    best_tree = None
    train_rmse = None
    best_test_rmse = float_info.max

    for iterations in range(niter):
        train_data, test_data, train_target, test_target = train_test_split(diabetes.data, diabetes.target, test_size=0.2)
        tree = DecisionTreeRegressor(min_samples_leaf=15)
        tree.fit(train_data, train_target)

        # Evaluate tree
        test_rmse = u.rmse(test_target, tree.predict(test_data))

        if (test_rmse < best_test_rmse):
            best_test_rmse = test_rmse
            best_tree = tree
            train_rmse = u.rmse(train_target, tree.predict(train_data))

    # Print RMSE of best tree
    print("Best decision tree:")
    u.print_results(train_rmse, best_test_rmse)

    if (dotfile != None):
        # Save visualisation
        export_graphviz(best_tree, max_depth=5, out_file=dotfile, rounded=True, filled=True, feature_names=["age", "sex", "BMI",
            "Average Blood Pressure", "Blood Serum 1", "Blood Serum 2", "Blood Serum 3", "Blood Serum 4", "Blood Serum 5", "Blood Serum 6"])
        print("Convert the .dot to .png using the command: dot -Tpng new.dot -o new.png")
