import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin


class MatrixLinearMixedEffectsModel(BaseEstimator, RegressorMixin):
    """
    Matrix linear model with mixed effects:

        Y_i = (A + A_g[i]) X_i + (B + B_g[i])

    where:
        - A: fixed slopes (n_outputs x n_features)
        - B: fixed intercepts (n_outputs,)
        - A_g: random slopes per group (G x n_outputs x n_features)
        - B_g: random intercepts per group (G x n_outputs)

    Fit all parameters by minimizing squared error.
    """

    def __init__(self):
        self.A_ = None  # fixed slopes (n_outputs x n_features)
        self.B_ = None  # fixed intercepts (n_outputs,)
        self.Ag_ = None  # random slopes (n_groups x n_outputs x n_features)
        self.Bg_ = None  # random intercepts (n_groups x n_outputs)
        self.groups_ = None

    def fit(self, X, Y, groups):
        X = np.asarray(X)
        Y = np.asarray(Y)
        groups = np.asarray(groups)

        n_samples, n_features = X.shape
        n_samples_y, n_outputs = Y.shape
        assert n_samples == n_samples_y, "X and Y must have same number of samples"
        assert len(groups) == n_samples, "Groups must be same length as samples"

        unique_groups, group_indices = np.unique(groups, return_inverse=True)
        n_groups = len(unique_groups)
        self.groups_ = unique_groups

        # Number of params:
        # fixed slopes: n_outputs*n_features
        # fixed intercepts: n_outputs
        # random slopes: n_groups*n_outputs*n_features
        # random intercepts: n_groups*n_outputs

        def pack_params(A, B, Ag, Bg):
            return np.concatenate(
                [
                    A.ravel(),
                    B.ravel(),
                    Ag.ravel(),
                    Bg.ravel(),
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
            return A, B, Ag, Bg

        # Objective: squared error loss
        def objective(params):
            A, B, Ag, Bg = unpack_params(params)
            # Calculate predictions for each sample
            preds = np.empty_like(Y)
            for i in range(n_samples):
                g = group_indices[i]
                # (n_outputs,) = (n_outputs x n_features) dot (n_features,) + (n_outputs,)
                preds[i] = (A + Ag[g]) @ X[i] + B + Bg[g]
            residuals = Y - preds
            return np.sum(residuals**2)

        # Init parameters: zeros
        A_init = np.zeros((n_outputs, n_features))
        B_init = np.zeros(n_outputs)
        Ag_init = np.zeros((n_groups, n_outputs, n_features))
        Bg_init = np.zeros((n_groups, n_outputs))
        init_params = pack_params(A_init, B_init, Ag_init, Bg_init)

        res = minimize(objective, init_params, method="L-BFGS-B")
        A_opt, B_opt, Ag_opt, Bg_opt = unpack_params(res.x)

        self.A_ = A_opt
        self.B_ = B_opt
        self.Ag_ = Ag_opt
        self.Bg_ = Bg_opt

        return self

    def predict(self, X, groups):
        X = np.asarray(X)
        groups = np.asarray(groups)
        n_samples, n_features = X.shape
        n_outputs = self.A_.shape[0]
        group_map = {g: i for i, g in enumerate(self.groups_)}

        preds = np.empty((n_samples, n_outputs))
        for i in range(n_samples):
            g = groups[i]
            g_idx = group_map[g]
            preds[i] = (self.A_ + self.Ag_[g_idx]) @ X[i] + self.B_ + self.Bg_[g_idx]
        return preds

    def score(self, X, Y, groups):
        Y = np.asarray(Y)
        Y_pred = self.predict(X, groups)
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        return 1 - ss_res / ss_tot
