import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt
import seaborn as sns

# ==== Family base class and example probit family ====


class Family:
    def link(self, mu):
        raise NotImplementedError

    def inv_link(self, eta):
        raise NotImplementedError

    def neg_log_likelihood(self, y, eta):
        raise NotImplementedError


class ProbitFamily(Family):
    from scipy.stats import norm

    def link(self, mu):
        # inverse CDF (probit)
        return self.norm.ppf(mu)

    def inv_link(self, eta):
        # CDF (normal)
        return self.norm.cdf(eta)

    def neg_log_likelihood(self, y, eta):
        # y are integers in 0,...,K-1, eta is latent variable
        # but here we don't use it for fitting directly, so can skip
        raise NotImplementedError("Use custom objective in model")


# ==== Ordinal Matrix Linear Mixed Effects Model ====


class OrdinalMatrixLinearMixedEffectsModel(BaseEstimator, RegressorMixin):
    def __init__(self, family=None, n_cats=None):
        """
        family: instance of Family class (e.g. ProbitFamily)
        n_cats: list or array of integers giving number of categories per output variable
                (each >=2)
        """
        self.family = family if family is not None else ProbitFamily()
        if n_cats is None:
            raise ValueError(
                "Must specify n_cats (list of number of categories per output)"
            )
        self.n_cats_ = np.array(n_cats, dtype=int)
        if np.any(self.n_cats_ < 2):
            raise ValueError("Each output must have at least 2 categories")

        self.A_ = None  # fixed slopes: (n_outputs, n_features)
        self.B_ = None  # fixed intercepts: (n_outputs,)
        self.Ag_ = None  # random slopes: (n_groups, n_outputs, n_features)
        self.Bg_ = None  # random intercepts: (n_groups, n_outputs)
        self.thresholds_ = None  # list of arrays (n_cats[j]-1 thresholds per output)
        self.groups_ = None

    def _pack_params(self, A, B, Ag, Bg, thresholds):
        parts = [A.ravel(), B.ravel(), Ag.ravel(), Bg.ravel()]
        for c in thresholds:
            parts.append(c)
        return np.concatenate(parts)

    def _unpack_params(self, params, n_groups, n_outputs, n_features):
        pos = 0
        A = params[pos : pos + n_outputs * n_features].reshape(n_outputs, n_features)
        pos += n_outputs * n_features

        B = params[pos : pos + n_outputs]
        pos += n_outputs

        Ag = params[pos : pos + n_groups * n_outputs * n_features].reshape(
            n_groups, n_outputs, n_features
        )
        pos += n_groups * n_outputs * n_features

        Bg = params[pos : pos + n_groups * n_outputs].reshape(n_groups, n_outputs)
        pos += n_groups * n_outputs

        thresholds = []
        for j in range(n_outputs):
            length = self.n_cats_[j] - 1
            c = params[pos : pos + length]
            pos += length
            thresholds.append(np.sort(c))  # ensure ordered thresholds

        return A, B, Ag, Bg, thresholds

    def fit(self, X, Y, groups):
        X = np.asarray(X)
        Y = np.asarray(Y, dtype=int)
        groups = np.asarray(groups)

        n_samples, n_features = X.shape
        n_samples_y, n_outputs = Y.shape
        assert n_samples == n_samples_y, "X and Y must have the same number of samples"
        assert len(groups) == n_samples, "Groups length must match samples"
        if n_outputs != len(self.n_cats_):
            raise ValueError("Length of n_cats must match number of outputs")

        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        self.groups_ = unique_groups

        # Objective function: negative log-likelihood of ordinal probit model with mixed effects
        def objective(params):
            A, B, Ag, Bg, thresholds = self._unpack_params(
                params, n_groups, n_outputs, n_features
            )
            total_nll = 0.0
            for i in range(n_samples):
                g = group_indices[i]
                x_i = X[i]
                y_i = Y[i]
                for j in range(n_outputs):
                    eta = (A[j] + Ag[g, j]) @ x_i + B[j] + Bg[g, j]
                    c = thresholds[j]
                    k = y_i[j]
                    c_lower = -np.inf if k == 0 else c[k - 1]
                    c_upper = np.inf if k == (len(c)) else c[k]
                    p = self.family.inv_link(c_upper - eta) - self.family.inv_link(
                        c_lower - eta
                    )
                    # Add small epsilon to prevent log(0)
                    p = max(p, 1e-12)
                    total_nll -= np.log(p)
            return total_nll

        # Initialize parameters
        A_init = np.zeros((n_outputs, n_features))
        B_init = np.zeros(n_outputs)
        Ag_init = np.zeros((n_groups, n_outputs, n_features))
        Bg_init = np.zeros((n_groups, n_outputs))

        # Initialize thresholds spaced evenly between -1 and 1 for each output
        thresholds_init = []
        for cats in self.n_cats_:
            if cats == 2:
                # only one threshold at 0
                thresholds_init.append(np.array([0.0]))
            else:
                thresholds_init.append(np.linspace(-1, 1, cats - 1))

        init_params = self._pack_params(
            A_init, B_init, Ag_init, Bg_init, thresholds_init
        )

        res = minimize(objective, init_params, method="L-BFGS-B")
        A_opt, B_opt, Ag_opt, Bg_opt, thresholds_opt = self._unpack_params(
            res.x, n_groups, n_outputs, n_features
        )

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

        proba_outputs = [
            np.zeros((n_samples, self.n_cats_[j])) for j in range(n_outputs)
        ]

        for i in range(n_samples):
            g = groups[i]
            g_idx = group_map[g]
            x_i = X[i]
            for j in range(n_outputs):
                eta = (
                    (self.A_[j] + self.Ag_[g_idx, j]) @ x_i
                    + self.B_[j]
                    + self.Bg_[g_idx, j]
                )
                c = self.thresholds_[j]
                k_cats = len(c) + 1
                for k in range(k_cats):
                    c_lower = -np.inf if k == 0 else c[k - 1]
                    c_upper = np.inf if k == k_cats - 1 else c[k]
                    p = self.family.inv_link(c_upper - eta) - self.family.inv_link(
                        c_lower - eta
                    )
                    proba_outputs[j][i, k] = p
        return proba_outputs

    def predict(self, X, groups):
        proba_outputs = self.predict_proba(X, groups)
        n_samples = X.shape[0]
        n_outputs = len(proba_outputs)
        y_pred = np.empty((n_samples, n_outputs), dtype=int)
        for j in range(n_outputs):
            y_pred[:, j] = np.argmax(proba_outputs[j], axis=1)
        return y_pred


# ==== Example Usage ====


def simulate_ordinal_data(
    n_samples=500, n_features=2, n_outputs=2, n_groups=3, n_cats=[3, 4], seed=42
):
    """
    Simulate data for ordinal matrix mixed model.
    n_cats: list of length n_outputs specifying number of categories per output.
    """
    rng = np.random.default_rng(seed)
    groups = rng.integers(0, n_groups, size=n_samples)

    X = rng.normal(size=(n_samples, n_features))
    A_true = rng.normal(scale=1.5, size=(n_outputs, n_features))
    B_true = rng.normal(scale=0.5, size=n_outputs)

    Ag_true = rng.normal(scale=0.5, size=(n_groups, n_outputs, n_features))
    Bg_true = rng.normal(scale=0.2, size=(n_groups, n_outputs))

    # Thresholds - increasing values for each output, shape = (n_cats[j]-1)
    thresholds_true = []
    for cats in n_cats:
        thresholds_true.append(np.linspace(-1.5, 1.5, cats - 1))

    Y = np.zeros((n_samples, n_outputs), dtype=int)

    def inv_link(eta):
        from scipy.stats import norm

        return norm.cdf(eta)

    for i in range(n_samples):
        g = groups[i]
        x_i = X[i]
        for j in range(n_outputs):
            eta = (A_true[j] + Ag_true[g, j]) @ x_i + B_true[j] + Bg_true[g, j]
            c = thresholds_true[j]
            k_cats = len(c) + 1
            # Compute category probabilities
            probs = np.zeros(k_cats)
            for k in range(k_cats):
                c_lower = -np.inf if k == 0 else c[k - 1]
                c_upper = np.inf if k == k_cats - 1 else c[k]
                probs[k] = inv_link(c_upper - eta) - inv_link(c_lower - eta)
            probs /= probs.sum()
            Y[i, j] = rng.choice(k_cats, p=probs)

    return X, Y, groups


def plot_results(X, Y, groups, model):
    import matplotlib.pyplot as plt
    import seaborn as sns

    n_outputs = Y.shape[1]
    n_features = X.shape[1]
    unique_groups = np.unique(groups)
    colors = sns.color_palette("tab10", len(unique_groups))
    group_color_map = {g: colors[i] for i, g in enumerate(unique_groups)}

    # 1) Plot predicted vs actual category counts heatmap per group and output
    Y_pred = model.predict(X, groups)

    fig, axes = plt.subplots(
        n_outputs,
        len(unique_groups),
        figsize=(4 * len(unique_groups), 4 * n_outputs),
        squeeze=False,
    )

    for j in range(n_outputs):
        n_cats = model.n_cats_[j]
        cats_labels = np.arange(n_cats)
        for g_idx, g in enumerate(unique_groups):
            mask = groups == g
            actual = Y[mask, j]
            pred = Y_pred[mask, j]
            # Count co-occurrences
            counts = np.zeros((n_cats, n_cats), dtype=int)
            for a, p in zip(actual, pred):
                counts[a, p] += 1
            ax = axes[j, g_idx]
            sns.heatmap(counts, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title(f"Output {j}, Group {g}")
            ax.set_xticklabels(cats_labels)
            ax.set_yticklabels(cats_labels, rotation=0)
    plt.tight_layout()
    plt.show()

    # 2) Plot predicted category distributions vs actual by group and output (histograms)
    fig, axes = plt.subplots(
        n_outputs,
        len(unique_groups),
        figsize=(4 * len(unique_groups), 3 * n_outputs),
        squeeze=False,
    )
    for j in range(n_outputs):
        n_cats = model.n_cats_[j]
        cats_labels = np.arange(n_cats)
        for g_idx, g in enumerate(unique_groups):
            mask = groups == g
            ax = axes[j, g_idx]
            sns.histplot(
                Y[mask, j],
                bins=np.arange(n_cats + 1) - 0.5,
                stat="probability",
                color="blue",
                label="Actual",
                ax=ax,
            )
            sns.histplot(
                Y_pred[mask, j],
                bins=np.arange(n_cats + 1) - 0.5,
                stat="probability",
                color="orange",
                label="Predicted",
                ax=ax,
            )
            ax.set_title(f"Output {j}, Group {g}")
            ax.set_xlabel("Category")
            ax.set_ylabel("Proportion")
            ax.legend()
    plt.tight_layout()
    plt.show()


def main():
    n_samples = 500
    n_features = 2
    n_outputs = 2
    n_groups = 3
    n_cats = [3, 4]  # different categories per output

    X, Y, groups = simulate_ordinal_data(
        n_samples, n_features, n_outputs, n_groups, n_cats
    )
    model = OrdinalMatrixLinearMixedEffectsModel(family=ProbitFamily(), n_cats=n_cats)

    print("Fitting model...")
    model.fit(X, Y, groups)

    print("Plotting results...")
    plot_results(X, Y, groups, model)


if __name__ == "__main__":
    main()
