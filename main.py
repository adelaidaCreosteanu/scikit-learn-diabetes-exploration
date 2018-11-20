import lin_reg
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
lin_reg.run_linear_regression(diabetes, 20)
