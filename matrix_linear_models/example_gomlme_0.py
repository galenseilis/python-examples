import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin

# --- Families ---


class OrdinalFamily:
    def link(self, eta, thresholds):
        raise NotImplementedError()

    def inv_link(self, eta, thresholds):
        raise NotImplementedError()

    def neg_log_likelihood(self, y, eta, thresholds):
        raise NotImplementedError()


class OrdinalLogisticFamily(OrdinalFamily):
    def inv_link(self, eta, thresholds):
        cdfs = 1 / (1 + np.exp(-(thresholds[None, :] - eta[:, None])))
        probs = np.empty((len(eta), len(thresholds) + 1))
        probs[:, 0] = cdfs[:, 0]
        for k in range(1, len(thresholds)):
            probs[:, k] = cdfs[:, k] - cdfs[:, k - 1]
        probs[:, -1] = 1 - cdfs[:, -1]
        return probs

    def neg_log_likelihood(self, y, eta, thresholds):
        probs = self.inv_link(eta, thresholds)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.sum(np.log(probs[np.arange(len(y)), y]))


class OrdinalProbitFamily(OrdinalFamily):
    def inv_link(self, eta, thresholds):
        cdfs = norm.cdf(thresholds[None, :] - eta[:, None])
        probs = np.empty((len(eta), len(thresholds) + 1))
        probs[:, 0] = cdfs[:, 0]
        for k in range(1, len(thresholds)):
            probs[:, k] = cdfs[:, k] - cdfs[:, k - 1]
        probs[:, -1] = 1 - cdfs[:, -1]
        return probs

    def neg_log_likelihood(self, y, eta, thresholds):
        probs = self.inv_link(eta, thresholds)
        probs = np.clip(probs, 1e-15, 1 - 1e-15)
        return -np.sum(np.log(probs[np.arange(len(y)), y]))


# --- Model ---


class OrdinalMatrixMixedEffectsModel(BaseEstimator, RegressorMixin):
    def __init__(self, n_thresholds, family=None):
        self.n_thresholds = n_thresholds
        self.family = family if family is not None else OrdinalLogisticFamily()

    def fit(self, X, Y, groups):
        X = np.asarray(X)
        Y = np.asarray(Y)
        groups = np.asarray(groups)

        n_samples, n_features = X.shape
        n_samples_y, n_outputs = Y.shape
        assert n_samples == n_samples_y
        assert len(groups) == n_samples

        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        self.groups_ = unique_groups

        # Initialize thresholds: sorted increasing
        # One threshold per split between categories, shared for all outputs
        # Thresholds shape: (n_thresholds,)
        thresh_init = np.linspace(-1.0, 1.0, self.n_thresholds)

        # Params:
        # fixed slopes: n_outputs * n_features
        # fixed intercepts: n_outputs
        # random slopes: n_groups * n_outputs * n_features
        # random intercepts: n_groups * n_outputs
        # thresholds: n_thresholds
        def pack_params(A, B, Ag, Bg, thresholds):
            return np.concatenate(
                [
                    A.ravel(),
                    B.ravel(),
                    Ag.ravel(),
                    Bg.ravel(),
                    thresholds,
                ]
            )

        def unpack_params(params):
            pos = 0
            A = params[pos : pos + n_outputs * n_features].reshape(
                n_outputs, n_features
            )
            pos += n_outputs * n_features
            B = params[pos : pos + n_outputs]
            pos += n_outputs
            Ag = params[pos : pos + n_groups * n_outputs * n_features].reshape(
                n_groups, n_outputs, n_features
            )
            pos += n_groups * n_outputs * n_features
            Bg = params[pos : pos + n_groups * n_outputs].reshape(n_groups, n_outputs)
            pos += n_groups * n_outputs
            thresholds = params[pos : pos + self.n_thresholds]
            return A, B, Ag, Bg, thresholds

        def objective(params):
            A, B, Ag, Bg, thresholds = unpack_params(params)
            # Enforce increasing thresholds during optimization (soft constraint)
            if not np.all(np.diff(thresholds) > 0):
                return 1e10

            eta = np.empty_like(Y, dtype=float)
            for i in range(n_samples):
                g = group_indices[i]
                eta[i] = (A + Ag[g]) @ X[i] + B + Bg[g]
            total_nll = 0
            for j in range(n_outputs):
                total_nll += self.family.neg_log_likelihood(
                    Y[:, j], eta[:, j], thresholds
                )
            return total_nll

        A_init = np.zeros((n_outputs, n_features))
        B_init = np.zeros(n_outputs)
        Ag_init = np.zeros((n_groups, n_outputs, n_features))
        Bg_init = np.zeros((n_groups, n_outputs))
        init_params = pack_params(A_init, B_init, Ag_init, Bg_init, thresh_init)

        res = minimize(objective, init_params, method="L-BFGS-B")
        A_opt, B_opt, Ag_opt, Bg_opt, thresholds_opt = unpack_params(res.x)

        self.A_ = A_opt
        self.B_ = B_opt
        self.Ag_ = Ag_opt
        self.Bg_ = Bg_opt
        self.thresholds_ = thresholds_opt

        return self

    def predict_proba(self, X, groups):
        X = np.asarray(X)
        groups = np.asarray(groups)
        n_samples, n_features = X.shape
        n_outputs = self.A_.shape[0]
        group_map = {g: i for i, g in enumerate(self.groups_)}

        eta = np.empty((n_samples, n_outputs))
        for i in range(n_samples):
            g = groups[i]
            g_idx = group_map[g]
            eta[i] = (self.A_ + self.Ag_[g_idx]) @ X[i] + self.B_ + self.Bg_[g_idx]

        probs = []
        for j in range(n_outputs):
            p = self.family.inv_link(eta[:, j], self.thresholds_)
            probs.append(p)
        # probs is list of (n_samples, n_cats) arrays, shape (n_outputs, n_samples, n_cats)
        return np.stack(probs, axis=1)

    def predict(self, X, groups):
        probs = self.predict_proba(X, groups)
        # Take argmax category
        preds = np.argmax(probs, axis=2)
        return preds

    def score(self, X, Y, groups):
        preds = self.predict(X, groups)
        return np.mean(preds == Y)


# --- Simulation & plotting ---


def simulate_ordinal_data(
    n_samples=300, n_features=2, n_outputs=2, n_groups=3, n_cats=4, seed=42
):
    np.random.seed(seed)
    X = np.random.uniform(-3, 3, (n_samples, n_features))
    groups = np.random.choice(n_groups, size=n_samples)
    true_A = np.array(
        [
            [[1.0, -0.5], [-0.3, 0.8]],  # group 0 slopes (n_outputs x n_features)
            [[1.2, -0.7], [-0.4, 0.5]],  # group 1
            [[0.8, -0.3], [-0.2, 0.9]],  # group 2
        ]
    )
    true_B = np.array([0.5, -0.5])  # fixed intercepts (n_outputs,)
    true_Bg = np.array(
        [
            [0.1, -0.1],
            [-0.2, 0.2],
            [0.0, 0.0],
        ]
    )
    thresholds = np.array([-1.0, 0.0, 1.0])

    eta = np.empty((n_samples, n_outputs))
    for i in range(n_samples):
        g = groups[i]
        eta[i] = true_A[g] @ X[i] + true_B + true_Bg[g]

    def category_probs(eta_row):
        n_thresholds = len(thresholds)
        n_outputs = len(eta_row)
        probs = np.empty((n_outputs, n_thresholds + 1))
        for j in range(n_outputs):
            cdfs = 1 / (1 + np.exp(-(thresholds - eta_row[j])))
            probs[j, 0] = cdfs[0]
            for k in range(1, n_thresholds):
                probs[j, k] = cdfs[k] - cdfs[k - 1]
            probs[j, -1] = 1 - cdfs[-1]
        return probs

    Y = np.empty((n_samples, n_outputs), dtype=int)
    for i in range(n_samples):
        p = category_probs(eta[i])
        for j in range(n_outputs):
            Y[i, j] = np.random.choice(len(thresholds) + 1, p=p[j])

    return X, Y, groups


def plot_results(X, Y, groups, model, family_name):
    n_samples, n_features = X.shape
    n_outputs = Y.shape[1]
    unique_groups = np.unique(groups)
    n_cats = model.thresholds_.size + 1

    colors = ["tab:blue", "tab:orange", "tab:green"]
    group_colors = {g: colors[i] for i, g in enumerate(unique_groups)}

    fig, axs = plt.subplots(
        n_outputs, n_features, figsize=(5 * n_features, 4 * n_outputs), squeeze=False
    )
    for out_i in range(n_outputs):
        for feat_i in range(n_features):
            ax = axs[out_i, feat_i]
            for g in unique_groups:
                idx = groups == g
                x_vals = X[idx, feat_i]
                y_vals = Y[idx, out_i]
                probs = model.predict_proba(X[idx], groups[idx])[:, out_i, :]
                # plot mean predicted category probability for that feature's values binned
                bins = np.linspace(np.min(x_vals), np.max(x_vals), 10)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])
                mean_probs = np.zeros((len(bin_centers), n_cats))
                for b in range(len(bin_centers)):
                    bin_mask = (x_vals >= bins[b]) & (x_vals < bins[b + 1])
                    if np.any(bin_mask):
                        mean_probs[b] = probs[bin_mask].mean(axis=0)
                    else:
                        mean_probs[b] = np.nan
                for cat in range(n_cats):
                    ax.plot(
                        bin_centers,
                        mean_probs[:, cat],
                        label=f"Cat {cat}" if g == unique_groups[0] else "",
                        color=group_colors[g],
                        alpha=0.7,
                        linestyle="-" if cat % 2 == 0 else "--",
                    )
                ax.scatter(
                    x_vals,
                    y_vals,
                    color=group_colors[g],
                    alpha=0.3,
                    label=f"Group {g}" if feat_i == 0 else "",
                )
            ax.set_xlabel(f"Feature {feat_i}")
            ax.set_ylabel(f"Output {out_i} category")
            if feat_i == 0:
                ax.legend(loc="upper left", fontsize=8)
            ax.set_title(f"{family_name} Family - Output {out_i}, Feature {feat_i}")
    plt.tight_layout()
    plt.show()

    # Heatmap of predicted vs actual per group per output
    for out_i in range(n_outputs):
        fig, axs = plt.subplots(
            1, len(unique_groups), figsize=(5 * len(unique_groups), 4)
        )
        for i, g in enumerate(unique_groups):
            ax = axs[i] if len(unique_groups) > 1 else axs
            idx = groups == g
            y_true = Y[idx, out_i]
            y_pred = model.predict(X[idx], groups[idx])[:, out_i]

            conf_mat = np.zeros((n_cats, n_cats))
            for t, p in zip(y_true, y_pred):
                conf_mat[t, p] += 1

            im = ax.imshow(conf_mat, cmap="viridis", origin="lower")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Group {g} Output {out_i}")
            fig.colorbar(im, ax=ax)
            ax.set_xticks(np.arange(n_cats))
            ax.set_yticks(np.arange(n_cats))
        plt.suptitle(f"{family_name} Family: Predicted vs Actual Counts")
        plt.tight_layout()
        plt.show()


# --- Run example ---

X, Y, groups = simulate_ordinal_data(
    n_samples=300, n_features=2, n_outputs=2, n_groups=3, n_cats=4
)

model = OrdinalMatrixMixedEffectsModel(n_thresholds=3, family=OrdinalLogisticFamily())
model.fit(X, Y, groups)

print("Training accuracy:", model.score(X, Y, groups))

plot_results(X, Y, groups, model, family_name="Logistic")
