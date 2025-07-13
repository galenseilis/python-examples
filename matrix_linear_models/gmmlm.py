import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, RegressorMixin

from scipy.special import gammaln


class Family:
    def link(self, mu):
        raise NotImplementedError

    def inverse_link(self, eta):
        raise NotImplementedError

    def neg_log_likelihood(self, y, mu):
        raise NotImplementedError


class NormalFamily(Family):
    def link(self, mu):
        return mu

    def inverse_link(self, eta):
        return eta

    def neg_log_likelihood(self, y, mu):
        return 0.5 * np.sum((y - mu) ** 2)


class PoissonFamily(Family):
    def link(self, mu):
        return np.log(np.clip(mu, 1e-10, None))

    def inverse_link(self, eta):
        return np.exp(eta)

    def neg_log_likelihood(self, y, mu):
        mu = np.clip(mu, 1e-10, None)
        return -np.sum(y * np.log(mu) - mu)


class NegativeBinomialFamily(Family):
    def __init__(self, size=1.0):
        self.size = size

    def link(self, mu):
        return np.log(np.clip(mu, 1e-10, None))

    def inverse_link(self, eta):
        return np.exp(eta)

    def neg_log_likelihood(self, y, mu):
        mu = np.clip(mu, 1e-10, None)
        r = self.size
        return -np.sum(
            gammaln(y + r)
            - gammaln(r)
            - gammaln(y + 1)
            + r * np.log(r / (r + mu))
            + y * np.log(mu / (r + mu))
        )


class MatrixLinearMixedEffectsModel(BaseEstimator, RegressorMixin):
    def __init__(self, family, lambda_re=0.0):
        self.family = family
        self.lambda_re = lambda_re
        self.A_ = None
        self.B_ = None
        self.Ag_ = None
        self.Bg_ = None
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

        # Better initialization using least squares in link space
        eta_init = self.family.link(Y.clip(min=1e-3))  # for Poisson/log link
        A_init = np.linalg.lstsq(X, eta_init, rcond=None)[0].T
        B_init = eta_init.mean(axis=0) - A_init @ X.mean(axis=0)

        Ag_init = np.zeros((n_groups, n_outputs, n_features))
        Bg_init = np.zeros((n_groups, n_outputs))
        init_params = pack_params(A_init, B_init, Ag_init, Bg_init)

        def objective(params):
            A, B, Ag, Bg = unpack_params(params)
            preds = np.empty_like(Y)
            for i in range(n_samples):
                g = group_indices[i]
                eta = (A + Ag[g]) @ X[i] + B + Bg[g]
                mu = self.family.inverse_link(eta)
                preds[i] = mu
            loss = self.family.neg_log_likelihood(Y, preds)
            reg = self.lambda_re * (np.sum(Ag**2) + np.sum(Bg**2))
            return loss + reg

        res = minimize(
            objective,
            init_params,
            method="L-BFGS-B",
            options={
                "maxiter": 1000,
                "gtol": 1e-6,
                "disp": True,
            },
        )

        if not res.success:
            print("⚠️ Optimization failed:", res.message)
        else:
            print("✅ Optimization succeeded:", res.message)

        A_opt, B_opt, Ag_opt, Bg_opt = unpack_params(res.x)
        self.A_ = A_opt
        self.B_ = B_opt
        self.Ag_ = Ag_opt
        self.Bg_ = Bg_opt
        return self

    def predict(self, X, groups):
        X = np.asarray(X)
        groups = np.asarray(groups)
        n_samples, _ = X.shape
        n_outputs = self.A_.shape[0]
        group_map = {g: i for i, g in enumerate(self.groups_)}
        preds = np.empty((n_samples, n_outputs))
        for i in range(n_samples):
            g = groups[i]
            g_idx = group_map[g]
            eta = (self.A_ + self.Ag_[g_idx]) @ X[i] + self.B_ + self.Bg_[g_idx]
            preds[i] = self.family.inverse_link(eta)
        return preds

    def score(self, X, Y, groups):
        Y_pred = self.predict(X, groups)
        ss_res = np.sum((Y - Y_pred) ** 2)
        ss_tot = np.sum((Y - Y.mean(axis=0)) ** 2)
        return 1 - ss_res / ss_tot
