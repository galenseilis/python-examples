import numpy as np
import matplotlib.pyplot as plt
from mmlm import MatrixLinearMixedEffectsModel

np.random.seed(1234)

# Parameters
n_samples_per_group = 50
n_features = 2
n_outputs = 2
n_groups = 3

# Fixed effects
A_fixed = np.array([[1.5, -0.5], [-1.0, 2.0]])
B_fixed = np.array([0.2, -0.5])

# Random slopes and intercepts per group
A_random = np.array(
    [[[0.0, 0.0], [0.0, 0.0]], [[1.0, -0.5], [-0.5, 1.0]], [[-1.5, 0.7], [1.0, -1.5]]]
)
B_random = np.array([[0.0, 0.0], [1.0, -1.0], [-2.0, 2.0]])

# Generate data
groups = np.repeat(np.arange(n_groups), n_samples_per_group)
X = np.empty((n_samples_per_group * n_groups, n_features))
for g in range(n_groups):
    mask = groups == g
    X[mask, 0] = np.linspace(0, 5, n_samples_per_group)
    X[mask, 1] = np.linspace(5, 0, n_samples_per_group)

noise_std = 0.3
Y = np.empty((n_samples_per_group * n_groups, n_outputs))
for i in range(len(groups)):
    g = groups[i]
    x_i = X[i]
    Y[i] = (
        (A_fixed + A_random[g]) @ x_i
        + B_fixed
        + B_random[g]
        + noise_std * np.random.randn(n_outputs)
    )

# Fit model
model = MatrixLinearMixedEffectsModel()
model.fit(X, Y, groups)
Y_pred = model.predict(X, groups)


# Fixed effect-only predictions (without group-specific deviations)
def fixed_only_predict(X):
    return (model.A_ @ X.T).T + model.B_


Y_fixed = fixed_only_predict(X)

# Plotting
fig, axs = plt.subplots(
    n_features, n_outputs, figsize=(14, 10), sharex="col", sharey="row"
)
colors = plt.cm.get_cmap("tab10", n_groups)

for i_feature in range(n_features):
    for j_output in range(n_outputs):
        ax = axs[i_feature, j_output]
        ax.set_title(f"Output {j_output + 1} vs Predictor {i_feature + 1}")
        ax.set_xlabel(f"Predictor {i_feature + 1} value")
        ax.set_ylabel(f"Output {j_output + 1}")

        # Plot group-specific points and predicted lines
        for g in range(n_groups):
            mask = groups == g
            x_vals = X[mask, i_feature]
            y_true_vals = Y[mask, j_output]
            y_pred_vals = Y_pred[mask, j_output]

            ax.scatter(
                x_vals,
                y_true_vals,
                alpha=0.6,
                label=f"Group {g} True",
                color=colors(g),
                marker="o",
            )

            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            y_pred_sorted = y_pred_vals[sort_idx]
            ax.plot(
                x_sorted, y_pred_sorted, color=colors(g), lw=2, label=f"Group {g} Pred"
            )

        # Plot fixed effect-only line
        x_range = np.linspace(X[:, i_feature].min(), X[:, i_feature].max(), 100)
        x_fixed = np.zeros((100, n_features))
        x_fixed[:, i_feature] = x_range  # vary only one predictor
        y_fixed_line = fixed_only_predict(x_fixed)[:, j_output]
        ax.plot(x_range, y_fixed_line, "k--", lw=2, label="Fixed effect")

        ax.grid(True)
        if i_feature == 0 and j_output == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()
