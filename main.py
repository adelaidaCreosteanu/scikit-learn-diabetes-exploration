import linear
import polynomial
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
linear.run_linear_regression(diabetes, 20)

polynomial.plot_poly(diabetes, [2])
