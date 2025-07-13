import numpy as np
import matplotlib.pyplot as plt

# The model class as defined earlier
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

class MatrixLinearModel(BaseEstimator, RegressorMixin):
    def __init__(self):
        self.A_ = None
        self.B_ = None

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        n_samples, n_features = X.shape
        n_samples_y, n_outputs = Y.shape
        assert n_samples == n_samples_y, "X and Y must have same number of samples"

        def pack_params(A, B):
            return np.concatenate([A.ravel(), B.ravel()])

        def unpack_params(params):
            A = params[:n_outputs * n_features].reshape(n_outputs, n_features)
            B = params[n_outputs * n_features:]
            return A, B

        def objective(params):
            A, B = unpack_params(params)
            Y_pred = X @ A.T + B
            residuals = Y - Y_pred
            return np.sum(residuals ** 2)

        init_A = np.zeros((n_outputs, n_features))
        init_B = np.zeros(n_outputs)
        init_params = pack_params(init_A, init_B)

        res = minimize(objective, init_params, method='L-BFGS-B')

        A_opt, B_opt = unpack_params(res.x)
        self.A_ = A_opt
        self.B_ = B_opt
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X @ self.A_.T + self.B_

    def score(self, X, Y):
        Y = np.asarray(Y)
        Y_pred = self.predict(X)
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        return 1 - ss_res / ss_tot


# ---- Generate synthetic data ----
np.random.seed(42)

n_samples = 200
n_features = 4
n_outputs = 3

# True parameters
A_true = np.array([[1.5, -2.0, 0.0, 0.5],
                   [0.0, 1.0, 2.0, -1.0],
                   [-1.0, 0.5, 1.0, 0.0]])
B_true = np.array([0.7, -1.2, 0.3])

# Input features X (shape: n_samples x n_features)
X = np.random.randn(n_samples, n_features)

# Generate outputs with noise
noise_std = 0.1
Y = X @ A_true.T + B_true + noise_std * np.random.randn(n_samples, n_outputs)

# ---- Fit our matrix linear model ----
model = MatrixLinearModel()
model.fit(X, Y)

print("True A:\n", A_true)
print("Learned A:\n", model.A_)
print("True B:", B_true)
print("Learned B:", model.B_)

# Predict on training data
Y_pred = model.predict(X)

# ---- Plot true vs predicted for each output dimension ----
fig, axs = plt.subplots(n_outputs, 1, figsize=(8, 4 * n_outputs), sharex=True)

for i in range(n_outputs):
    ax = axs[i]
    ax.scatter(Y[:, i], Y_pred[:, i], alpha=0.7)
    lims = [
        np.min([Y[:, i], Y_pred[:, i]]),  # min of true and pred
        np.max([Y[:, i], Y_pred[:, i]]),  # max of true and pred
    ]
    ax.plot(lims, lims, 'r--', label="Ideal")
    ax.set_title(f"Output dimension {i+1}: True vs Predicted")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
