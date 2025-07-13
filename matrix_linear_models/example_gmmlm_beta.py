import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from gmmlm import MatrixLinearMixedEffectsModel

# Define the BetaFamily as shown above
from scipy.special import gammaln


class BetaFamily:
    def __init__(self, phi=5.0):
        self.phi = phi

    def link(self, mu):
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        return np.log(mu / (1 - mu))

    def inverse_link(self, eta):
        return 1 / (1 + np.exp(-eta))

    def neg_log_likelihood(self, y, mu):
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        alpha = mu * self.phi
        beta = (1 - mu) * self.phi
        return -np.sum(
            gammaln(alpha + beta)
            - gammaln(alpha)
            - gammaln(beta)
            + (alpha - 1) * np.log(y)
            + (beta - 1) * np.log(1 - y)
        )


# Simulated data
np.random.seed(777)
n_per_group = 50
groups = np.array(["X"] * n_per_group + ["Y"] * n_per_group)
n_samples = len(groups)

X = np.zeros((n_samples, 2))
X[:, 0] = np.linspace(-2, 2, n_samples)  # Trend
X[:, 1] = np.random.randn(n_samples)  # Noise

A_true = np.array([[1.0, -0.5], [-1.0, 0.3]])
B_true = np.array([0.0, 0.2])

Ag_true = {
    "X": np.array([[0.5, 0.0], [-0.5, 0.0]]),
    "Y": np.array([[-0.5, 0.0], [0.5, 0.0]]),
}
Bg_true = {"X": np.array([0.2, -0.3]), "Y": np.array([-0.2, 0.3])}

# Generate Y in (0, 1)
Y = np.zeros((n_samples, 2))
for i in range(n_samples):
    g = groups[i]
    eta = (A_true + Ag_true[g]) @ X[i] + B_true + Bg_true[g]
    mu = 1 / (1 + np.exp(-eta))
    alpha = mu * 5
    beta = (1 - mu) * 5
    Y[i] = np.random.beta(alpha, beta)

# Standardize X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit model
family = BetaFamily(phi=5.0)
model = MatrixLinearMixedEffectsModel(family=family)
model.fit(X_scaled, Y, groups)
Y_pred = model.predict(X_scaled, groups)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
colors = {"X": "darkgreen", "Y": "darkred"}

for i in range(2):  # predictors
    for j in range(2):  # predicted targets
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
        ax.set_ylabel(f"Y{j + 1} (Proportion)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()
