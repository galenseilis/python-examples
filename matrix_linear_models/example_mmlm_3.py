import numpy as np
import matplotlib.pyplot as plt
from mmlm import MatrixLinearMixedEffectsModel

np.random.seed(1234)

# Parameters
n_samples_per_group = 50
n_features = 2  # two predictors
n_outputs = 2  # two outputs
n_groups = 3

# Fixed effects: shape (n_outputs x n_features)
A_fixed = np.array(
    [
        [1.5, -0.5],  # slopes for output 0
        [-1.0, 2.0],  # slopes for output 1
    ]
)

# Fixed intercepts shape (n_outputs,)
B_fixed = np.array([0.2, -0.5])

# Random slopes per group: shape (n_groups x n_outputs x n_features)
A_random = np.array(
    [
        [[0.0, 0.0], [0.0, 0.0]],  # group 0
        [[1.0, -0.5], [-0.5, 1.0]],  # group 1
        [[-1.5, 0.7], [1.0, -1.5]],  # group 2
    ]
)

# Random intercepts per group: shape (n_groups x n_outputs)
B_random = np.array([[0.0, 0.0], [1.0, -1.0], [-2.0, 2.0]])

# Generate groups and data
groups = np.repeat(np.arange(n_groups), n_samples_per_group)
X = np.empty((n_samples_per_group * n_groups, n_features))

# For each group, generate predictor values in a grid range for visualization
for g in range(n_groups):
    mask = groups == g
    # Generate predictor 0 and 1 values randomly in [0,5]
    X[mask, 0] = np.linspace(0, 5, n_samples_per_group)
    X[mask, 1] = np.linspace(5, 0, n_samples_per_group)  # reversed for variety

# Generate outputs with noise
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

# Predict
Y_pred = model.predict(X, groups)

print("Fixed slopes (A):\n", model.A_)
print("Fixed intercepts (B):\n", model.B_)
print(
    "Random slopes per group (Ag):\n",
    model.Ag_.reshape(n_groups, n_outputs, n_features),
)
print("Random intercepts per group (Bg):\n", model.Bg_)

# Plot setup: 2x2 grid, rows = predictors, columns = outputs
fig, axs = plt.subplots(
    n_features, n_outputs, figsize=(14, 10), sharex="col", sharey="row"
)

colors = plt.cm.get_cmap("tab10", n_groups)

for i_feature in range(n_features):
    for j_output in range(n_outputs):
        ax = axs[i_feature, j_output]
        ax.set_title(f"Output {j_output + 1} vs Predictor {i_feature + 1}")
        ax.set_xlabel(f"Predictor {i_feature + 1} value")
        ax.set_ylabel(f"Output {j_output + 1} value")

        for g in range(n_groups):
            mask = groups == g
            x_vals = X[mask, i_feature]
            y_true_vals = Y[mask, j_output]
            y_pred_vals = Y_pred[mask, j_output]

            # Scatter true points
            ax.scatter(
                x_vals,
                y_true_vals,
                alpha=0.6,
                label=f"Group {g} True",
                color=colors(g),
                marker="o",
            )

            # Sort for smooth line plot
            sort_idx = np.argsort(x_vals)
            x_sorted = x_vals[sort_idx]
            y_pred_sorted = y_pred_vals[sort_idx]

            # Plot predicted regression line
            ax.plot(
                x_sorted,
                y_pred_sorted,
                color=colors(g),
                linewidth=2,
                label=f"Group {g} Predicted",
            )

        ax.grid(True)
        if i_feature == 0 and j_output == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()
