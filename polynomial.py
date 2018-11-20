import utility as u
from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt


def perform_pca(data):
    pca = KernelPCA(n_components=4, kernel='rbf', gamma=0.02)
    transformed = pca.fit_transform(data)
    return transformed


def run_poly_reg(diabetes, degrees, with_pca=True):
    x = diabetes.data
    y = diabetes.target

    if (with_pca):
        x = perform_pca(diabetes.data)

    for d in degrees:
        train_rmse_sum = 0
        test_rmse_sum = 0

        for _ in range(20):
            poly_reg = Pipeline((
                ("poly_features", PolynomialFeatures(degree=d, include_bias=True)),
                ("sgd_reg", LinearRegression()),
            ))

            # Uniformly randomly split data 80% training and 20% testing
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            poly_reg.fit(x_train, y_train)

            # Add RMSE
            train_rmse_sum += u.rmse(y_train, poly_reg.predict(x_train))
            test_rmse_sum += u.rmse(y_test, poly_reg.predict(x_test))
        
        print("Degree ", d)
        u.print_results(train_rmse_sum/20, test_rmse_sum/20)


def plot_poly(diabetes, degrees, with_pca=True):
    x = diabetes.data
    y = diabetes.target

    if (with_pca):
        x = perform_pca(diabetes.data)
    
    # Uniformly randomly split data 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    # Create figure where all three models will be plotted
    plt.figure()
    colours = ['blue', 'yellowgreen', 'gold']

    # Create subplots with the first 4 features and plot the ground truth
    for sub_fig in range(1,5):
        plt.subplot(220+sub_fig)
        plt.ylim(-100,350)
        plt.ylabel("Response of interest")
        plt.xlabel("Feature " + str(sub_fig))
        plt.scatter(x_test[:,sub_fig-1], y_test, color='black', label="ground truth", marker='+')

    for count, d in enumerate(degrees):
        # Polynomially expand every feature of the data:
        polynomial_regression = Pipeline((
            ("poly_features", PolynomialFeatures(degree=d, include_bias=True)),
            ("sgd_reg", LinearRegression()),
        ))
        polynomial_regression.fit(x_train, y_train)

        test_predict = polynomial_regression.predict(x_test)

        for sub_fig in range(1,5):
            plt.subplot(220+sub_fig)
            plt.scatter(x_test[:,sub_fig-1], test_predict, color=colours[count], label="d %d" % d, marker='*')

    plt.legend(loc='lower right')
    plt.show()
