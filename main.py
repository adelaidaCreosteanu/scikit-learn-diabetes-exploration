import linear
import polynomial
import lasso
import decisionTree
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

print("===Linear regression===")
linear.run_linear_regression(diabetes, 20)

print("===Polynomial regression===")
polynomial.plot_poly(diabetes, [2])

print("===Lasso regularisation===")
lasso.plot_alphas_lasso(diabetes)
print("Optimal alpha: ", lasso.optimal_alpha(diabetes))

print("===Decision tree===")
decisionTree.best_tree(diabetes, dotfile="new.dot")
