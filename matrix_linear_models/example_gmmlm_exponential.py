import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Exponential distribution family
class ExponentialFamily:
    @staticmethod
    def link(mu):
        mu = np.clip(mu, 1e-6, None)
        return np.log(mu)

    @staticmethod
    def inverse_link(eta):
        return np.exp(eta)

    @staticmethod
    def neg_log_likelihood(y, mu):
        mu = np.clip(mu, 1e-6, None)
        return np.sum(np.log(mu) + y / mu)


from gmmlm import MatrixLinearMixedEffectsModel

# Simulate non-stationary time-series-like data
np.random.seed(456)
n_per_group = 50
groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)
n_samples = len(groups)

# Predictor 0: time (non-stationary trend), Predictor 1: noise
X = np.zeros((n_samples, 2))
X[:, 0] = np.linspace(0, 10, n_samples)  # time
X[:, 1] = np.random.randn(n_samples)  # noise

# True fixed effects (base trend)
A_true = np.array([[0.25, 0.0], [0.10, 0.0]])
B_true = np.array([0.0, 0.0])

# Group-specific effects
Ag_true = {
    "A": np.array([[0.1, 0.0], [0.1, 0.0]]),
    "B": np.array([[-0.1, 0.0], [-0.1, 0.0]]),
}
Bg_true = {"A": np.array([0.2, 0.5]), "B": np.array([-0.2, -0.5])}

# Response Y: time until event (exponential)
Y = np.zeros((n_samples, 2))
for i in range(n_samples):
    g = groups[i]
    eta = (A_true + Ag_true[g]) @ X[i] + B_true + Bg_true[g]
    mu = np.exp(eta)  # mean of exponential
    Y[i] = np.random.exponential(scale=mu)

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit model
family = ExponentialFamily()
model = MatrixLinearMixedEffectsModel(family=family)
model.fit(X_scaled, Y, groups)
Y_pred = model.predict(X_scaled, groups)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
colors = {"A": "blue", "B": "orange"}

for i in range(2):  # predictors
    for j in range(2):  # predicted variables
        ax = axes[i, j]
        for g in np.unique(groups):
            mask = groups == g
            ax.scatter(
                X_scaled[mask, i],
                Y[mask, j],
                color=colors[g],
                alpha=0.5,
                label=f"{g} True" if i == 0 and j == 0 else "",
            )
            ax.scatter(
                X_scaled[mask, i],
                Y_pred[mask, j],
                color=colors[g],
                marker="x",
                alpha=0.8,
                label=f"{g} Pred" if i == 0 and j == 0 else "",
            )
        ax.set_title(f"Predictor {i + 1} vs Target {j + 1}")
        ax.set_xlabel(f"X{i + 1}")
        ax.set_ylabel(f"Y{j + 1} (Time)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()
