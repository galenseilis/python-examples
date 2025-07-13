import numpy as np
import matplotlib.pyplot as plt
from mmlm import MatrixLinearMixedEffectsModel

np.random.seed(42)

# Setup
n_samples_per_group = 50
n_groups = 3
n_outputs = 2
n_raw_features = 2  # the original predictors before expansion
groups = np.repeat(np.arange(n_groups), n_samples_per_group)

# === Generate raw features ===
X_raw = np.empty((n_samples_per_group * n_groups, n_raw_features))
for g in range(n_groups):
    mask = groups == g
    X_raw[mask, 0] = np.linspace(0, 2 * np.pi, n_samples_per_group)
    X_raw[mask, 1] = np.linspace(0, 4 * np.pi, n_samples_per_group)


# === Apply Fourier feature expansion ===
def fourier_features(X_raw):
    # For each feature: add sin(x), cos(x), sin(2x), cos(2x)
    X_new = [X_raw]
    for i in range(X_raw.shape[1]):
        x = X_raw[:, i]
        X_new.append(np.sin(x))
        X_new.append(np.cos(x))
        X_new.append(np.sin(2 * x))
        X_new.append(np.cos(2 * x))
    return np.column_stack(X_new)


X = fourier_features(X_raw)

# === True coefficients for fixed + random effects ===
n_features = X.shape[1]

# Fixed effects: matrix (n_outputs x n_features)
A_fixed = np.random.randn(n_outputs, n_features) * 0.5
B_fixed = np.array([1.0, -1.5])

# Random effects: per group (n_groups x n_outputs x n_features)
A_random = np.random.randn(n_groups, n_outputs, n_features) * 0.2
B_random = np.random.randn(n_groups, n_outputs) * 0.5

# === Generate response ===
noise_std = 0.3
Y = np.empty((X.shape[0], n_outputs))
for i in range(X.shape[0]):
    g = groups[i]
    xi = X[i]
    Y[i] = (
        (A_fixed + A_random[g]) @ xi
        + B_fixed
        + B_random[g]
        + noise_std * np.random.randn(n_outputs)
    )

# === Fit the model ===
model = MatrixLinearMixedEffectsModel()
model.fit(X, Y, groups)
Y_pred = model.predict(X, groups)

# === Plotting ===
fig, axs = plt.subplots(
    n_raw_features, n_outputs, figsize=(14, 10), sharex="col", sharey="row"
)
colors = plt.cm.get_cmap("tab10", n_groups)

for i_feat in range(n_raw_features):
    for j_out in range(n_outputs):
        ax = axs[i_feat, j_out]
        ax.set_title(f"Output {j_out + 1} vs Predictor {i_feat + 1}")
        ax.set_xlabel(f"Raw Predictor {i_feat + 1}")
        ax.set_ylabel(f"Output {j_out + 1}")

        for g in range(n_groups):
            mask = groups == g
            x_vals = X_raw[mask, i_feat]
            y_true = Y[mask, j_out]
            y_pred = Y_pred[mask, j_out]

            sorted_idx = np.argsort(x_vals)
            x_sorted = x_vals[sorted_idx]
            y_true_sorted = y_true[sorted_idx]
            y_pred_sorted = y_pred[sorted_idx]

            # True points
            ax.scatter(
                x_sorted,
                y_true_sorted,
                alpha=0.6,
                color=colors(g),
                label=f"Group {g}" if j_out == 0 and i_feat == 0 else None,
            )
            # Predicted line
            ax.plot(x_sorted, y_pred_sorted, color=colors(g), lw=2)

        ax.grid(True)

# Add legend outside the first subplot
axs[0, 0].legend(loc="upper left", bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()
