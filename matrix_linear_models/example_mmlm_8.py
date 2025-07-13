import numpy as np
import matplotlib.pyplot as plt
from mmlm import MatrixLinearMixedEffectsModel

np.random.seed(42)

# Settings
n_groups = 3
samples_per_group = 100
n_outputs = 2

# Total samples
n_samples = n_groups * samples_per_group
groups = np.repeat(np.arange(n_groups), samples_per_group)

# Generate time predictor (same for all groups for simplicity)
time = np.linspace(0, 20, samples_per_group)
time_all = np.tile(time, n_groups)

# Generate cumulative sum of step functions for each group:
# For each group, create some random step changes (+1 or -1) at random time points
step_cumsum_all = np.zeros(n_samples)
for g in range(n_groups):
    idx_start = g * samples_per_group
    idx_end = idx_start + samples_per_group
    steps = np.zeros(samples_per_group)
    # Randomly choose 3 step points
    step_points = np.sort(
        np.random.choice(samples_per_group - 1, size=3, replace=False)
    )
    current = 0
    for i in range(samples_per_group):
        if i in step_points:
            current += np.random.choice([-1, 1])
        steps[i] = current
    step_cumsum_all[idx_start:idx_end] = steps

# Combine predictors: shape (n_samples, 2)
X = np.column_stack([time_all, step_cumsum_all])

# Create true coefficients for fixed + random effects
n_features = X.shape[1]
A_fixed = np.array(
    [
        [0.5, -0.2],  # output1 slopes for time and step_cumsum
        [0.3, 0.7],
    ]
)  # output2 slopes for time and step_cumsum
B_fixed = np.array([1.0, -1.0])  # intercepts for output1 and output2

# Random effects: per group (n_groups x n_outputs x n_features)
A_random = np.random.randn(n_groups, n_outputs, n_features) * 0.1
B_random = np.random.randn(n_groups, n_outputs) * 0.5

# Generate responses
noise_std = 0.3
Y = np.empty((n_samples, n_outputs))
for i in range(n_samples):
    g = groups[i]
    xi = X[i]
    Y[i] = (
        (A_fixed + A_random[g]) @ xi
        + B_fixed
        + B_random[g]
        + noise_std * np.random.randn(n_outputs)
    )

# Fit the model
model = MatrixLinearMixedEffectsModel()
model.fit(X, Y, groups)
Y_pred = model.predict(X, groups)

# Plotting
fig, axs = plt.subplots(2, n_outputs, figsize=(14, 8))
colors = plt.cm.get_cmap("tab10", n_groups)

predictor_names = ["Time", "Step Cumulative Sum"]
output_names = ["Output 1", "Output 2"]

for i_feat in range(2):
    for j_out in range(n_outputs):
        ax = axs[i_feat, j_out]
        ax.set_title(f"{output_names[j_out]} vs {predictor_names[i_feat]}")
        ax.set_xlabel(predictor_names[i_feat])
        ax.set_ylabel(output_names[j_out])

        for g in range(n_groups):
            mask = groups == g
            x_vals = X[mask, i_feat]
            y_true = Y[mask, j_out]
            y_pred = Y_pred[mask, j_out]

            # Sort for better plotting lines
            sorted_idx = np.argsort(x_vals)
            x_sorted = x_vals[sorted_idx]
            y_true_sorted = y_true[sorted_idx]
            y_pred_sorted = y_pred[sorted_idx]

            ax.scatter(
                x_sorted,
                y_true_sorted,
                alpha=0.5,
                color=colors(g),
                label=f"Group {g}" if i_feat == 0 and j_out == 0 else None,
            )
            ax.plot(x_sorted, y_pred_sorted, color=colors(g), linewidth=2)

        ax.grid(True)

# Add legend outside the top-left plot
axs[0, 0].legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
