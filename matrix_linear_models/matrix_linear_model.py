import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin


class MatrixLinearModel(BaseEstimator, RegressorMixin):
    """
    Linear model generalized to matrix equations: Y = A X + B
    where
    - X: shape (n_samples, n_features)
    - Y: shape (n_samples, n_outputs)
    - A: (n_outputs, n_features)
    - B: (n_outputs,) bias vector

    Fits A and B by minimizing squared error via scipy optimizer.
    """

    def __init__(self):
        self.A_ = None  # shape (n_outputs, n_features)
        self.B_ = None  # shape (n_outputs,)

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples, n_features = X.shape
        n_samples_y, n_outputs = Y.shape
        assert n_samples == n_samples_y, "X and Y must have same number of samples"

        # Flatten parameters: first A (n_outputs*n_features), then B (n_outputs)
        def pack_params(A, B):
            return np.concatenate([A.ravel(), B.ravel()])

        def unpack_params(params):
            A = params[: n_outputs * n_features].reshape(n_outputs, n_features)
            B = params[n_outputs * n_features :]
            return A, B

        # Objective: squared error loss
        def objective(params):
            A, B = unpack_params(params)
            Y_pred = X @ A.T + B  # shape (n_samples, n_outputs)
            residuals = Y - Y_pred
            return np.sum(residuals**2)

        # Initialize A, B with zeros
        init_A = np.zeros((n_outputs, n_features))
        init_B = np.zeros(n_outputs)
        init_params = pack_params(init_A, init_B)

        # Optimize
        res = minimize(objective, init_params, method="L-BFGS-B")

        A_opt, B_opt = unpack_params(res.x)
        self.A_ = A_opt
        self.B_ = B_opt
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.A_.T + self.B_

    def score(self, X, Y):
        """R^2 score"""
        Y = np.asarray(Y)
        Y_pred = self.predict(X)
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        return 1 - ss_res / ss_tot
