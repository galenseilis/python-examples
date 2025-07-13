import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


class NormalFamily:
    @staticmethod
    def link(mu):
        return mu

    @staticmethod
    def inverse_link(eta):
        return eta

    @staticmethod
    def neg_log_likelihood(y, mu):
        return np.sum((y - mu) ** 2)


from gmmlm import MatrixLinearMixedEffectsModel

# Seed for reproducibility
np.random.seed(789)

n_per_group = 50
groups = np.array(["Group1"] * n_per_group + ["Group2"] * n_per_group)
n_samples = len(groups)

X = np.random.randn(n_samples, 2)

# True fixed effects (slopes and intercepts)
A_true = np.array([[1.0, -0.5], [0.5, 0.3]])
B_true = np.array([0.5, -0.2])

# Amplified random effects for groups (strong differences)
Ag_true = {
    "Group1": np.array([[1.0, 0.5], [-0.5, 1.0]]),
    "Group2": np.array([[-1.0, -0.5], [0.5, -1.0]]),
}

Bg_true = {"Group1": np.array([2.0, -1.5]), "Group2": np.array([-2.0, 1.5])}

# Generate response with Gaussian noise
Y = np.zeros((n_samples, 2))
sigma = 0.5

for i in range(n_samples):
    g = groups[i]
    eta = (A_true + Ag_true[g]) @ X[i] + B_true + Bg_true[g]
    mu = eta
    Y[i] = mu + np.random.randn(2) * sigma

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit the model
normal_family = NormalFamily()
model = MatrixLinearMixedEffectsModel(family=normal_family)
model.fit(X_scaled, Y, groups)
Y_pred = model.predict(X_scaled, groups)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
colors = {"Group1": "blue", "Group2": "red"}

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
        ax.set_ylabel(f"Y{j + 1}")

# Legend once for all plots
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()
