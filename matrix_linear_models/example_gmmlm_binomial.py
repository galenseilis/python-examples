import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# Binomial family for logistic regression
class BinomialFamily:
    @staticmethod
    def link(mu):
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        return np.log(mu / (1 - mu))

    @staticmethod
    def inverse_link(eta):
        return 1 / (1 + np.exp(-eta))

    @staticmethod
    def neg_log_likelihood(y, mu):
        mu = np.clip(mu, 1e-6, 1 - 1e-6)
        return -np.sum(y * np.log(mu) + (1 - y) * np.log(1 - mu))


# Import your model
from gmmlm import MatrixLinearMixedEffectsModel

# Simulate data
np.random.seed(321)
n_per_group = 50
groups = np.array(["C1"] * n_per_group + ["C2"] * n_per_group)
n_samples = len(groups)

X = np.random.randn(n_samples, 2)

# Fixed effects
A_true = np.array([[2.0, -1.0], [-1.0, 1.0]])
B_true = np.array([0.0, 0.5])

# Random effects
Ag_true = {
    "C1": np.array([[1.0, -0.5], [0.5, -1.0]]),
    "C2": np.array([[-1.0, 0.5], [-0.5, 1.0]]),
}
Bg_true = {"C1": np.array([0.5, -0.5]), "C2": np.array([-0.5, 0.5])}

# Binary outcomes
Y = np.zeros((n_samples, 2))

for i in range(n_samples):
    g = groups[i]
    eta = (A_true + Ag_true[g]) @ X[i] + B_true + Bg_true[g]
    p = 1 / (1 + np.exp(-eta))
    Y[i] = np.random.binomial(1, p)

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the model
binomial_family = BinomialFamily()
model = MatrixLinearMixedEffectsModel(family=binomial_family)
model.fit(X_scaled, Y, groups)
Y_pred_prob = model.predict(X_scaled, groups)

# Plot
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
colors = {"C1": "purple", "C2": "green"}

for i in range(2):  # predictors
    for j in range(2):  # predicted outputs
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
                Y_pred_prob[mask, j],
                color=colors[g],
                marker="x",
                alpha=0.8,
                label=f"{g} Pred" if i == 0 and j == 0 else "",
            )
        ax.set_title(f"Predictor {i + 1} vs Target {j + 1}")
        ax.set_xlabel(f"X{i + 1}")
        ax.set_ylabel(f"Y{j + 1} (Prob)")

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()
