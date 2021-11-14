import sklearn.utils.fixes
from numpy.ma import MaskedArray

sklearn.utils.fixes.MaskedArray = MaskedArray

import time
import numpy as np


from sklearn.svm import SVR
from sklearn.preprocessing import scale
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import SGDRegressor, BayesianRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
)
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    cross_val_score,
    ShuffleSplit,
)


def evaluate_model(reg, x, y):
    cv = ShuffleSplit(n_splits=3)
    cv_mse = cross_val_score(reg, x, y, cv=cv, scoring="neg_mean_squared_error")
    cv_mae = cross_val_score(reg, x, y, cv=cv, scoring="neg_mean_absolute_error")
    cv_r2 = cross_val_score(reg, x, y, cv=cv, scoring="r2")
    return (np.mean(cv_mse), np.mean(cv_mae), np.mean(cv_r2))


def sgd(x, y, scale):
    x = np.array(x)
    y = np.array(y)

    params = {
        "loss": "squared_loss",
        "max_iter": 10 ** 7,
        "tol": 0.0000001,
        "penalty": "l2",
        "l1_ratio": 0.15,
        "epsilon": 0.01,
        "learning_rate": "invscaling",
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    reg = SGDRegressor(**params)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1

    score = str(reg.score(list(x_test), y_test))
    print(
        "stochastic gradient descent score:   " + str(score) + " time: " + str(time_el)
    )

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def gradient_boost_reg(x, y, scale):
    params = {
        "loss": "ls",
        "n_estimators": 2000,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "criterion": "mse",
        "max_depth": 10,
        "tol": 0.0001,
    }

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    reg = GradientBoostingRegressor(**params)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print(
        "gradient boost score:                " + str(score) + " time: " + str(time_el)
    )

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def random_forest(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    params = {
        "max_depth": 20,
        "n_estimators": 500,
        "bootstrap": True,
        "min_samples_leaf": 2,
        "n_jobs": 16,
        "verbose": False,
        "n_jobs": 4,
    }

    reg = RandomForestRegressor(**params)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print(
        "random forest score:                 " + str(score) + " time: " + str(time_el)
    )

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def extra_trees(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    params = {
        "n_estimators": 500,
        "min_samples_split": 2,
        "min_samples_leaf": 2,
        "n_jobs": 16,
        "verbose": False,
    }

    reg = ExtraTreesRegressor(**params)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print("Extra trees score:                 " + str(score) + " time: " + str(time_el))

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def gaussian(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    kernel = DotProduct() + WhiteKernel()
    reg = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, random_state=0)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print(
        "gaussian process score:              " + str(score) + " time: " + str(time_el)
    )

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)
    return reg


def kernel(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    # reg = KernelRidge(alpha=0.0001, degree = 10,kernel = "polynomial")
    reg = KernelRidge(kernel="rbf", alpha=0.00005, gamma=0.0001)

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print(
        "kernel regression score:             " + str(score) + " time: " + str(time_el)
    )

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)
    return reg


def bayesian(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    reg = BayesianRidge(
        n_iter=10000,
        tol=1e-7,
        copy_X=True,
        alpha_1=1e-03,
        alpha_2=1e-03,
        lambda_1=1e-03,
        lambda_2=1e-03,
    )

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1
    score = reg.score(list(x_test), y_test)
    print(
        "bayesian score:                      " + str(score) + " time: " + str(time_el)
    )

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg


def svr(x, y, scale):
    # change C
    # scale data
    # L1/L2 normalization

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    svr_rbf = SVR(kernel="rbf", C=0.001, gamma=0.1, epsilon=0.1, cache_size=4000)
    est_rbf = svr_rbf
    svr_lin = SVR(kernel="linear", C=0.1, gamma="auto", cache_size=4000)
    est_lin = svr_lin
    svr_poly = SVR(
        kernel="poly",
        C=100,
        gamma="auto",
        degree=6,
        epsilon=0.1,
        coef0=0.5,
        cache_size=4000,
    )
    est_poly = svr_poly

    t1 = time.time()
    est_rbf.fit(list(x_train), y_train)
    t2 = time.time()
    time_rbf = t2 - t1
    s1 = svr_rbf.score(list(x_test), y_test)

    t1 = time.time()
    est_lin.fit(list(x_train), y_train)
    t2 = time.time()
    time_svr = t2 - t1
    s2 = svr_lin.score(list(x_test), y_test)

    t1 = time.time()
    est_poly.fit(list(x_train), y_train)
    t2 = time.time()
    time_poly = t2 - t1
    score = svr_poly.score(list(x_test), y_test)

    print("linear svr score:                    " + str(s2) + " time: " + str(time_rbf))
    print("radial basis svr score:              " + str(s1) + " time: " + str(time_svr))
    print(
        "polynomial svr score:                "
        + str(score)
        + " time: "
        + str(time_poly)
    )

    score = str(mean_squared_error(svr_poly.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_poly))

    score = str(mean_absolute_error(svr_poly.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_poly))

    score = str(r2_score(svr_poly.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_poly))

    score_mae = mean_absolute_error(svr_poly.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return svr_poly


def sk_nn(x, y, scale):
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    reg = MLPRegressor(
        random_state=1,
        max_iter=100000,
        learning_rate_init=0.00001,
        learning_rate="adaptive",
        early_stopping=True,
        tol=1e-7,
        shuffle=True,
        solver="adam",
        activation="relu",
        hidden_layer_sizes=(1000,),
        verbose=False,
        alpha=0.00001,
    )

    t1 = time.time()
    reg.fit(list(x_train), y_train)
    t2 = time.time()
    time_el = t2 - t1

    score = reg.score(list(x_test), y_test)
    print(
        "Neural Network score:                " + str(score) + " time: " + str(time_el)
    )

    score = str(mean_squared_error(reg.predict(x_test), y_test))
    print("MSE score:   " + str(score) + " time: " + str(time_el))

    score = str(mean_absolute_error(reg.predict(x_test), y_test))
    print("MAE score:   " + str(score) + " time: " + str(time_el))

    score = str(r2_score(reg.predict(x_test), y_test))
    print("r2 score:   " + str(score) + " time: " + str(time_el))

    score_mae = mean_absolute_error(reg.predict(x_test), y_test)
    print("scaled MAE")
    print(scale * score_mae)

    return reg
