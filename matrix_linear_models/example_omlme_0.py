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
