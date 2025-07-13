import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from omlme import OrdinalMatrixLinearMixedEffectsModel


def simulate_ordinal_data(
    n_samples=300,
    n_features=2,
    n_outputs=2,
    n_groups=3,
    n_cats_per_output=[4, 5],
    seed=42,
):
    """
    Simulate ordinal data with possibly different numbers of categories per output.
    """
    np.random.seed(seed)
    X = np.random.uniform(-3, 3, (n_samples, n_features))
    groups = np.random.choice(n_groups, size=n_samples)

    # True parameters
    true_A = np.array(
        [
            [[1.0, -0.5], [-0.3, 0.8]],
            [[1.2, -0.7], [-0.4, 0.5]],
            [[0.8, -0.3], [-0.2, 0.9]],
        ]
    )  # (n_groups x n_outputs x n_features)
    true_B = np.array([0.5, -0.5])  # (n_outputs,)
    true_Bg = np.array(
        [
            [0.1, -0.1],
            [-0.2, 0.2],
            [0.0, 0.0],
        ]
    )  # (n_groups x n_outputs)

    # Thresholds: list, one array per output, spaced evenly around zero
    thresholds = [np.linspace(-1.5, 1.5, nc - 1) for nc in n_cats_per_output]

    # Compute linear predictors
    eta = np.empty((n_samples, n_outputs))
    for i in range(n_samples):
        g = groups[i]
        eta[i] = true_A[g] @ X[i] + true_B + true_Bg[g]

    def category_probs(eta_row):
        # eta_row shape: (n_outputs,)
        probs = []
        for j in range(n_outputs):
            t = thresholds[j]
            cdfs = 1 / (1 + np.exp(-(t - eta_row[j])))
            p = np.empty(len(t) + 1)
            p[0] = cdfs[0]
            for k in range(1, len(t)):
                p[k] = cdfs[k] - cdfs[k - 1]
            p[-1] = 1 - cdfs[-1]
            probs.append(p)
        return np.array(probs)  # (n_outputs x n_cats_j)

    # Sample Y from multinomial with probs per output
    Y = np.empty((n_samples, n_outputs), dtype=int)
    for i in range(n_samples):
        p = category_probs(eta[i])
        for j in range(n_outputs):
            Y[i, j] = np.random.choice(len(p[j]), p=p[j])

    return X, Y, groups, thresholds


# Simulate data
X, Y, groups, thresholds = simulate_ordinal_data()

# Initialize and fit model with user thresholds as initial values for fitting
model = OrdinalMatrixLinearMixedEffectsModel(
    thresholds_init=thresholds, max_iter=200, reg_lambda=1e-2
)
model.fit(X, Y, groups)

# Predict
Y_pred = model.predict(X, groups)  # predicted category indices

# Plotting

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

n_outputs = Y.shape[1]
n_features = X.shape[1]
unique_groups = np.unique(groups)
colors = sns.color_palette("Set2", n_colors=len(unique_groups))

fig, axes = plt.subplots(
    n_features, n_outputs, figsize=(4 * n_outputs, 3 * n_features), sharex=True
)

for i_feat in range(n_features):
    for j_out in range(n_outputs):
        ax = axes[i_feat, j_out] if n_features > 1 else axes[j_out]

        for g, c in zip(unique_groups, colors):
            mask = groups == g
            ax.scatter(
                X[mask, i_feat],
                Y_pred[mask, j_out],
                alpha=0.6,
                label=f"Group {g}",
                color=c,
                s=15,
            )

        ax.set_xlabel(f"Predictor {i_feat}")
        ax.set_ylabel(f"Predicted Category {j_out}")
        if i_feat == 0 and j_out == 0:
            ax.legend(loc="upper right")

plt.tight_layout()
plt.show()
