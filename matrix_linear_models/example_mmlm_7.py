import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from mmlm import MatrixLinearMixedEffectsModel

np.random.seed(123)

# Settings
n_categories_1 = 4
n_categories_2 = 3
n_groups = 3
n_outputs = 2
samples_per_group = 60

# Total samples
n_samples = n_groups * samples_per_group
groups = np.repeat(np.arange(n_groups), samples_per_group)

# === Generate categorical data ===
cat1 = np.random.choice(n_categories_1, size=n_samples)
cat2 = np.random.choice(n_categories_2, size=n_samples)

# === One-hot encode categories ===
encoder = OneHotEncoder(sparse_output=False, categories="auto", drop=None)
X_cat = np.column_stack([cat1, cat2])
X_encoded = encoder.fit_transform(X_cat)

# Track original category labels for plotting
cat1_labels = encoder.categories_[0]
cat2_labels = encoder.categories_[1]

# === Create synthetic model ===
n_features = X_encoded.shape[1]
A_fixed = np.random.randn(n_outputs, n_features) * 0.5
B_fixed = np.array([0.5, -1.0])
A_random = np.random.randn(n_groups, n_outputs, n_features) * 0.2
B_random = np.random.randn(n_groups, n_outputs) * 0.5

noise_std = 0.2
Y = np.empty((n_samples, n_outputs))
for i in range(n_samples):
    g = groups[i]
    xi = X_encoded[i]
    Y[i] = (
        (A_fixed + A_random[g]) @ xi
        + B_fixed
        + B_random[g]
        + noise_std * np.random.randn(n_outputs)
    )

# === Fit model ===
model = MatrixLinearMixedEffectsModel()
model.fit(X_encoded, Y, groups)
Y_pred = model.predict(X_encoded, groups)

# === Plotting ===
fig, axs = plt.subplots(2, n_outputs, figsize=(14, 8))
colors = plt.cm.get_cmap("tab10", n_groups)

for i_cat, (cat, label_list, cat_name) in enumerate(
    zip([cat1, cat2], [cat1_labels, cat2_labels], ["Cat1", "Cat2"])
):
    for j_out in range(n_outputs):
        ax = axs[i_cat, j_out]
        ax.set_title(f"Output {j_out + 1} vs {cat_name}")
        ax.set_xlabel(f"{cat_name} Category")
        ax.set_ylabel(f"Output {j_out + 1}")

        for g in range(n_groups):
            mask = groups == g
            x_vals = cat[mask]
            y_true = Y[mask, j_out]
            y_pred = Y_pred[mask, j_out]

            # Plot true values and mean predictions per category
            for c in np.unique(cat):
                c_mask = x_vals == c
                x_pos = c + (g - 1) * 0.1  # shift for visual clarity
                if np.any(c_mask):
                    # True values (scatter)
                    ax.scatter(
                        [x_pos] * np.sum(c_mask),
                        y_true[c_mask],
                        alpha=0.5,
                        color=colors(g),
                        label=f"Group {g}" if c == 0 else None,
                    )
                    # Mean predicted value (cross)
                    y_pred_mean = np.mean(y_pred[c_mask])
                    ax.plot([x_pos], [y_pred_mean], "x", color=colors(g), markersize=10)

        ax.set_xticks(np.arange(len(label_list)))
        ax.set_xticklabels(label_list)
        ax.grid(True)
        if i_cat == 0 and j_out == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1))

plt.tight_layout()
plt.show()
