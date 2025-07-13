import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


# === Define Negative Binomial Family ===
class NegativeBinomialFamily:
    def __init__(self, r=5.0):  # dispersion (number of failures)
        self.r = r

    def link(self, mu):
        return np.log(mu)

    def inverse_link(self, eta):
        return np.exp(eta)

    def neg_log_likelihood(self, y, mu):
        from scipy.special import gammaln

        r = self.r
        mu = np.clip(mu, 1e-6, None)
        return -np.sum(
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1)
            + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu))
        )


# === Assume model class is available from local module ===
from gmmlm import MatrixLinearMixedEffectsModel

# === Generate synthetic data ===
np.random.seed(123)
n_per_group = 50
groups = np.array(["A"] * n_per_group + ["B"] * n_per_group)
n_samples = len(groups)

X = np.random.randn(n_samples, 2)

# True coefficients
A_true = np.array([[0.6, -0.4], [0.3, 0.2]])
B_true = np.array([1.0, 0.8])
Ag_true = {
    "A": np.array([[0.1, 0.2], [-0.2, 0.1]]),
    "B": np.array([[-0.1, 0.1], [0.3, -0.3]]),
}
Bg_true = {"A": np.array([0.2, -0.1]), "B": np.array([-0.3, 0.1])}


# Generate Negative Binomial outcomes
def sample_negative_binomial(mu, r):
    p = r / (r + mu)
    return np.random.negative_binomial(r, p)


Y = np.zeros((n_samples, 2))
r_nb = 5.0  # dispersion parameter

for i in range(n_samples):
    g = groups[i]
    eta = (A_true + Ag_true[g]) @ X[i] + B_true + Bg_true[g]
    mu = np.exp(eta)
    Y[i] = [sample_negative_binomial(m, r_nb) for m in mu]

# Standardize predictors
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Fit model ===
nb_family = NegativeBinomialFamily(r=r_nb)
model = MatrixLinearMixedEffectsModel(family=nb_family)
model.fit(X_scaled, Y, groups)
Y_pred = model.predict(X_scaled, groups)

# === Plot results ===
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
colors = {"A": "orange", "B": "purple"}

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

# Add a shared legend
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper right")
plt.tight_layout()
plt.show()
