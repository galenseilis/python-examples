import numpy as np
import matplotlib.pyplot as plt
from mmlm import MatrixLinearMixedEffectsModel

np.random.seed(123)

# Parameters
n_samples_per_group = 50
n_features = 1  # single feature for easy plotting
n_outputs = 2  # two outputs
n_groups = 3

# Fixed effects (shape: n_outputs x n_features)
A_fixed = np.array(
    [
        [2.0],  # output 1 slope
        [-1.0],
    ]
)  # output 2 slope

# Fixed intercepts (shape: n_outputs)
B_fixed = np.array([0.5, 1.0])

# Random effects (shape: n_groups x n_outputs x n_features) for slopes
A_random = np.array(
    [
        [[0.0], [0.0]],  # group 0 random slopes
        [[1.0], [-0.5]],  # group 1 random slopes
        [[-1.5], [1.0]],  # group 2 random slopes
    ]
)

# Random intercepts (shape: n_groups x n_outputs)
B_random = np.array(
    [
        [0.0, 0.0],  # group 0 random intercepts
        [1.0, -1.0],  # group 1 random intercepts
        [-2.0, 2.0],  # group 2 random intercepts
    ]
)

# Create data
groups = np.repeat(np.arange(n_groups), n_samples_per_group)
X = np.empty((n_samples_per_group * n_groups, n_features))

# For plotting, use evenly spaced X per group
for g in range(n_groups):
    mask = groups == g
    X[mask, 0] = np.linspace(0, 5, n_samples_per_group)

# Generate outputs Y with noise
noise_std = 0.2
Y = np.empty((n_samples_per_group * n_groups, n_outputs))

for i in range(len(groups)):
    g = groups[i]
    x_i = X[i]
    # Y[i] shape: (n_outputs,)
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

# Plot true points and predicted regression lines
colors = plt.cm.get_cmap("tab10", n_groups)
fig, axs = plt.subplots(n_outputs, 1, figsize=(10, 6 * n_outputs), sharex=True)

for output_idx in range(n_outputs):
    ax = axs[output_idx]
    ax.set_title(f"Output variable {output_idx + 1}")
    ax.set_xlabel("Feature X")
    ax.set_ylabel(f"Y_{output_idx + 1}")

    for g in range(n_groups):
        mask = groups == g
        x_group = X[mask, 0]
        y_true_group = Y[mask, output_idx]
        y_pred_group = Y_pred[mask, output_idx]

        # Scatter true points
        ax.scatter(
            x_group, y_true_group, alpha=0.6, label=f"Group {g} True", color=colors(g)
        )

        # Sort for line plot
        sorted_idx = np.argsort(x_group)
        x_sorted = x_group[sorted_idx]
        y_pred_sorted = y_pred_group[sorted_idx]

        # Plot predicted regression line
        ax.plot(
            x_sorted,
            y_pred_sorted,
            color=colors(g),
            linewidth=2,
            label=f"Group {g} Predicted",
        )

    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()
