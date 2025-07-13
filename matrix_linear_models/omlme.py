import numpy as np
from scipy.optimize import minimize
from scipy.special import expit  # sigmoid
from sklearn.base import BaseEstimator


class OrdinalMatrixMixedEffectsModel(BaseEstimator):
    def __init__(self, n_categories):
        self.n_categories = n_categories
        self.A_ = None
        self.B_ = None
        self.Ag_ = None
        self.Bg_ = None
        self.thresholds_ = None
        self.groups_ = None

    def _initialize_thresholds(self):
        # Use evenly spaced thresholds, strictly increasing
        return np.linspace(-1, 1, self.n_categories - 1)

    def _pack_params(self, A, B, Ag, Bg, thresholds):
        return np.concatenate(
            [
                A.ravel(),
                B.ravel(),
                Ag.ravel(),
                Bg.ravel(),
                thresholds.ravel(),
            ]
        )

    def _unpack_params(self, params, n_outputs, n_features, n_groups):
        pos = 0
        A = params[pos : pos + n_outputs * n_features].reshape(n_outputs, n_features)
        pos += n_outputs * n_features
        B = params[pos : pos + n_outputs]
        pos += n_outputs
        Ag = params[pos : pos + n_groups * n_outputs * n_features].reshape(
            n_groups, n_outputs, n_features
        )
        pos += n_groups * n_outputs * n_features
        Bg = params[pos : pos + n_groups * n_outputs].reshape(n_groups, n_outputs)
        pos += n_groups * n_outputs
        thresholds = params[pos:].reshape(self.n_categories - 1)
        return A, B, Ag, Bg, thresholds

    def _ordinal_log_likelihood(self, y, eta, thresholds):
        """Negative log-likelihood for ordinal logistic regression"""
        n = len(y)
        ll = 0.0
        for i in range(n):
            k = int(y[i])
            eta_i = eta[i]
            if k == 0:
                prob = expit(thresholds[0] - eta_i)
            elif k == self.n_categories - 1:
                prob = 1 - expit(thresholds[-1] - eta_i)
            else:
                prob = expit(thresholds[k] - eta_i) - expit(thresholds[k - 1] - eta_i)
            prob = np.clip(prob, 1e-12, 1 - 1e-12)
            ll -= np.log(prob)
        return ll

    def fit(self, X, Y, groups):
        X = np.asarray(X)
        Y = np.asarray(Y).astype(int)
        groups = np.asarray(groups)

        n_samples, n_features = X.shape
        n_outputs = 1  # ordinal targets are scalar
        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        self.groups_ = unique_groups

        # Initialize parameters
        A = np.zeros((n_outputs, n_features))
        B = np.zeros(n_outputs)
        Ag = np.zeros((n_groups, n_outputs, n_features))
        Bg = np.zeros((n_groups, n_outputs))
        thresholds = self._initialize_thresholds()

        def objective(params):
            A, B, Ag, Bg, thresholds = self._unpack_params(
                params, n_outputs, n_features, n_groups
            )
            # Ensure thresholds are strictly increasing
            if not np.all(np.diff(thresholds) > 0):
                return 1e6  # large penalty for invalid thresholds

            eta = np.zeros(n_samples)
            for i in range(n_samples):
                g = group_indices[i]
                eta[i] = (A + Ag[g]) @ X[i] + B + Bg[g]
            return self._ordinal_log_likelihood(Y, eta, thresholds)

        # Flatten all params into a vector
        init_params = self._pack_params(A, B, Ag, Bg, thresholds)

        result = minimize(objective, init_params, method="L-BFGS-B")
        A_opt, B_opt, Ag_opt, Bg_opt, thresholds_opt = self._unpack_params(
            result.x, n_outputs, n_features, n_groups
        )

        self.A_ = A_opt
        self.B_ = B_opt
        self.Ag_ = Ag_opt
        self.Bg_ = Bg_opt
        self.thresholds_ = thresholds_opt
        return self

    def predict_proba(self, X, groups):
        X = np.asarray(X)
        groups = np.asarray(groups)
        n_samples = X.shape[0]
        group_map = {g: i for i, g in enumerate(self.groups_)}
        eta = np.zeros(n_samples)
        for i in range(n_samples):
            g = groups[i]
            g_idx = group_map[g]
            eta[i] = (self.A_ + self.Ag_[g_idx]) @ X[i] + self.B_ + self.Bg_[g_idx]
        probs = np.zeros((n_samples, self.n_categories))
        for i in range(n_samples):
            t = self.thresholds_
            e = eta[i]
            cdf = np.concatenate(([0], expit(t - e), [1]))
            probs[i] = np.diff(cdf)
        return probs

    def predict(self, X, groups):
        probas = self.predict_proba(X, groups)
        return np.argmax(probas, axis=1)


import numpy as np
import matplotlib.pyplot as plt
from omlme import OrdinalMatrixMixedEffectsModel

np.random.seed(42)

# === 1. Generate synthetic ordinal data ===

n_samples_per_group = 100
groups = np.repeat(["A", "B"], n_samples_per_group)
n_samples = len(groups)

X = np.random.randn(n_samples, 2)

# Define true fixed effects
A_true = np.array([[1.0, -1.0]])
B_true = np.array([0.5])

# Random effects per group
Ag_true = {"A": np.array([[0.5, -0.2]]), "B": np.array([[-0.3, 0.6]])}
Bg_true = {"A": np.array([0.2]), "B": np.array([-0.4])}

# Thresholds (for 4 categories: 0–3)
thresholds = np.array([-0.5, 0.5, 1.5])

# Generate latent scores and ordinal responses
eta = np.zeros(n_samples)
y = np.zeros(n_samples, dtype=int)

for i in range(n_samples):
    g = groups[i]
    x_i = X[i]
    eta_i = (A_true + Ag_true[g]) @ x_i + B_true + Bg_true[g]
    eta[i] = eta_i
    # Compute category via thresholds
    if eta_i < thresholds[0]:
        y[i] = 0
    elif eta_i < thresholds[1]:
        y[i] = 1
    elif eta_i < thresholds[2]:
        y[i] = 2
    else:
        y[i] = 3

# === 2. Fit model ===

model = OrdinalMatrixMixedEffectsModel(n_categories=4)
model.fit(X, y, groups)
y_pred = model.predict(X, groups)

# === 3. Plotting: Scatter plots of true vs predicted by predictor and group ===

colors = {"A": "steelblue", "B": "darkorange"}
group_labels = np.unique(groups)

fig1, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for j, predictor_idx in enumerate([0, 1]):
    ax = axes[j]
    for group in group_labels:
        mask = groups == group
        ax.scatter(
            X[mask, predictor_idx],
            y[mask],
            label=f"Group {group} (true)",
            alpha=0.6,
            color=colors[group],
            marker="o",
        )
        ax.scatter(
            X[mask, predictor_idx],
            y_pred[mask],
            label=f"Group {group} (pred)",
            alpha=0.4,
            color=colors[group],
            marker="x",
        )
    ax.set_xlabel(f"Predictor {predictor_idx + 1}")
    ax.set_ylabel("Ordinal Category")
    ax.set_title(f"Ordinal Response vs Predictor {predictor_idx + 1}")
    ax.grid(True)

handles, labels = axes[0].get_legend_handles_labels()
fig1.legend(handles, labels, loc="lower center", ncol=2)
plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.suptitle("Ordinal Mixed Effects Model: True vs Predicted Categories", fontsize=14)

# === 4. Plotting: Heatmaps of predicted vs actual category counts by group ===

fig2, axes = plt.subplots(
    1, len(group_labels), figsize=(6 * len(group_labels), 5), sharey=True
)

if len(group_labels) == 1:
    axes = [axes]

n_categories = 4
for i, group in enumerate(group_labels):
    mask = groups == group
    actual = y[mask]
    pred = y_pred[mask]

    # Compute confusion matrix: rows=actual, cols=predicted
    conf_mat = np.zeros((n_categories, n_categories), dtype=int)
    for a, p in zip(actual, pred):
        conf_mat[a, p] += 1

    ax = axes[i]
    im = ax.imshow(conf_mat, cmap="Blues", origin="lower")

    ax.set_xticks(np.arange(n_categories))
    ax.set_yticks(np.arange(n_categories))
    ax.set_xlabel("Predicted Category")
    if i == 0:
        ax.set_ylabel("Actual Category")
    ax.set_title(f"Group {group}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.suptitle("Predicted vs Actual Category Counts by Group", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])

plt.show()
