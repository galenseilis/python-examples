import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Define Gamma family
class GammaFamily:
    @staticmethod
    def link(mu):
        return np.log(mu)

    @staticmethod
    def inverse_link(eta):
        return np.exp(eta)

    @staticmethod
    def neg_log_likelihood(y, mu):
        y = np.clip(y, 1e-6, None)
        mu = np.clip(mu, 1e-6, None)
        return np.sum((y - mu) ** 2 / mu**2)


# Import your custom model
from gmmlm import MatrixLinearMixedEffectsModel

# Simulate data
np.random.seed(123)
n_per_group = 50
groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)
n_samples = len(groups)

X = np.random.randn(n_samples, 2)

# Coefficients
A_true = np.array([[0.6, -0.4], [0.3, 0.2]])
B_true = np.array([0.5, 0.4])
Ag_true = {
    "A": np.array([[0.1, 0.1], [-0.1, 0.2]]),
    "B": np.array([[-0.1, 0.2], [0.1, -0.2]]),
}
Bg_true = {"A": np.array([0.1, -0.1]), "B": np.array([-0.2, 0.2])}

# Generate Gamma-distributed responses
Y = np.zeros((n_samples, 2))
shape_param = 2.0  # constant shape (k), scale = mu / k

for i in range(n_samples):
    g = groups[i]
    eta = (A_true + Ag_true[g]) @ X[i] + B_true + Bg_true[g]
    mu = np.exp(eta)
    Y[i] = np.random.gamma(shape=shape_param, scale=mu / shape_param)

# Standardize X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit model
gamma_family = GammaFamily()
model = MatrixLinearMixedEffectsModel(family=gamma_family)
model.fit(X_scaled, Y, groups)
Y_pred = model.predict(X_scaled, groups)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
colors = {"A": "crimson", "B": "teal"}

for i in range(2):  # predictors
    for j in range(2):  # predicted variables
        ax = axes[i, j]
        for g in np.unique(groups):
            mask = groups == g
            ax.scatter(
                X_scaled[mask, i],
                Y[mask, j],
                color=colors[g],
                alpha=0.4,
                label=f"{g} True" if i == 0 and j == 0 else "",
            )
            ax.scatter(
                X_scaled[mask, i],
                Y_pred[mask, j],
                color=colors[g],
                marker="x",
                alpha=0.7,
                label=f"{g} Pred" if i == 0 and j == 0 else "",
            )
        ax.set_title(f"Predictor {i + 1} vs Target {j + 1}")
        ax.set_xlabel(f"X{i + 1}")
        ax.set_ylabel(f"Y{j + 1}")

# Shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()
