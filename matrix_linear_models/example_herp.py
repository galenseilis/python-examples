import numpy as np
import matplotlib.pyplot as plt
from herp import OrdinalMatrixLinearMixedEffectsModel, LogitFamily

# === Simulated data generator ===

def simulate_ordinal_data(n_samples=500, n_features=2, n_outputs=2, n_groups=3, n_cats_list=[3, 4]):
    rng = np.random.default_rng(42)
    X = rng.normal(size=(n_samples, n_features))
    groups = rng.integers(0, n_groups, size=n_samples)

    A = rng.normal(scale=1.0, size=(n_outputs, n_features))
    B = rng.normal(scale=1.0, size=n_outputs)
    Ag = rng.normal(scale=0.5, size=(n_groups, n_outputs, n_features))
    Bg = rng.normal(scale=0.5, size=(n_groups, n_outputs))

    thresholds = [np.sort(rng.uniform(-2, 2, size=n_cats_list[j] - 1)) for j in range(n_outputs)]
    inv_link = LogitFamily().inv_link

    Y = np.zeros((n_samples, n_outputs), dtype=int)

    for i in range(n_samples):
        x_i = X[i]
        g = groups[i]
        for j in range(n_outputs):
            eta = A[j] @ x_i + B[j] + Ag[g, j] @ x_i + Bg[g, j]
            t = thresholds[j]
            probs = np.zeros(len(t) + 1)
            for k in range(len(t) + 1):
                lo = -np.inf if k == 0 else t[k - 1]
                hi = np.inf if k == len(t) else t[k]
                probs[k] = inv_link(hi - eta) - inv_link(lo - eta)
            probs /= probs.sum()
            Y[i, j] = rng.choice(len(probs), p=probs)
    return X, Y, groups

# === Plotting ===

def plot_predictions(Y_true, Y_pred, groups, n_outputs):
    fig, axes = plt.subplots(1, n_outputs, figsize=(6 * n_outputs, 5))
    if n_outputs == 1:
        axes = [axes]

    for j in range(n_outputs):
        ax = axes[j]
        ax.scatter(Y_true[:, j], Y_pred[:, j], c=groups, cmap="tab10", alpha=0.6)
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"Output {j}")
        ax.set_xticks(np.unique(Y_true[:, j]))
        ax.set_yticks(np.unique(Y_pred[:, j]))
    plt.suptitle("Predicted vs True (by color group)")
    plt.tight_layout()
    plt.show()

def plot_heatmaps(Y_true, Y_pred, groups, n_outputs, n_groups):
    for j in range(n_outputs):
        fig, axs = plt.subplots(1, n_groups, figsize=(4 * n_groups, 4))
        fig.suptitle(f"Heatmaps for Output {j}")
        for g in range(n_groups):
            ax = axs[g]
            mask = groups == g
            yt = Y_true[mask, j]
            yp = Y_pred[mask, j]
            max_cat = max(yt.max(), yp.max()) + 1
            heat = np.zeros((max_cat, max_cat), dtype=int)
            for t, p in zip(yt, yp):
                heat[t, p] += 1
            im = ax.imshow(heat, cmap='Blues')
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(f"Group {g}")
            fig.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.show()

# === Main script ===

def main():
    X, Y, groups = simulate_ordinal_data(n_samples=500, n_features=2, n_outputs=2, n_groups=3, n_cats_list=[3, 4])
    
    model = OrdinalMatrixLinearMixedEffectsModel(
        family=LogitFamily(),
        n_cats=[3, 4],
        random_slopes=True,
        random_intercepts=True,
        alpha_l1=0.01,
        alpha_l2=0.01
    )

    print("Fitting model...")
    model.fit(X, Y, groups)
    
    Y_pred = model.predict(X, groups)

    plot_predictions(Y, Y_pred, groups, n_outputs=2)
    plot_heatmaps(Y, Y_pred, groups, n_outputs=2, n_groups=3)

if __name__ == "__main__":
    main()
