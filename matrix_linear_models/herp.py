import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator

# === FAMILY CLASSES ===

class OrdinalFamily:
    def link(self, x):
        raise NotImplementedError

    def inv_link(self, x):
        raise NotImplementedError

    def name(self):
        return self.__class__.__name__

class LogitFamily(OrdinalFamily):
    def link(self, x):
        return np.log(x / (1 - x))

    def inv_link(self, x):
        return expit(x)

class ProbitFamily(OrdinalFamily):
    def link(self, x):
        from scipy.stats import norm
        return norm.ppf(x)

    def inv_link(self, x):
        from scipy.stats import norm
        return norm.cdf(x)

# === MODEL ===

class OrdinalMatrixLinearMixedEffectsModel(BaseEstimator):
    def __init__(self, family=None, n_cats=None,
                 random_intercepts=True,
                 random_slopes=True,
                 alpha_l1=0.0,
                 alpha_l2=0.0):
        self.family = family or ProbitFamily()
        self.n_cats = n_cats
        self.random_intercepts = random_intercepts
        self.random_slopes = random_slopes
        self.alpha_l1 = alpha_l1
        self.alpha_l2 = alpha_l2

    def _pack_params(self, A, B, Ag, Bg, thresholds):
        return np.concatenate([
            A.ravel(),
            B.ravel(),
            Ag.ravel() if Ag is not None else np.array([]),
            Bg.ravel() if Bg is not None else np.array([]),
            np.concatenate([t for t in thresholds])
        ])

    def _unpack_params(self, params):
        pos = 0
        A = params[pos:pos + self.n_outputs * self.n_features].reshape(self.n_outputs, self.n_features)
        pos += A.size
        B = params[pos:pos + self.n_outputs]
        pos += B.size

        if self.random_slopes:
            Ag = params[pos:pos + self.n_groups * self.n_outputs * self.n_features].reshape(
                self.n_groups, self.n_outputs, self.n_features)
            pos += Ag.size
        else:
            Ag = None

        if self.random_intercepts:
            Bg = params[pos:pos + self.n_groups * self.n_outputs].reshape(self.n_groups, self.n_outputs)
            pos += Bg.size
        else:
            Bg = None

        thresholds = []
        for nc in self.n_cats:
            t = params[pos:pos + nc - 1]
            thresholds.append(t)
            pos += len(t)

        return A, B, Ag, Bg, thresholds

    def _negative_log_likelihood(self, params, X, Y, group_idx):
        A, B, Ag, Bg, thresholds = self._unpack_params(params)
        inv_link = self.family.inv_link
        loss = 0.0

        for i in range(X.shape[0]):
            x_i = X[i]
            g = group_idx[i]

            for j in range(self.n_outputs):
                eta = A[j] @ x_i + B[j]
                if self.random_slopes:
                    eta += Ag[g, j] @ x_i
                if self.random_intercepts:
                    eta += Bg[g, j]

                t = thresholds[j]
                k = Y[i, j]
                k_cats = len(t) + 1

                if k == 0:
                    p = inv_link(t[0] - eta)
                elif k == k_cats - 1:
                    p = 1 - inv_link(t[-1] - eta)
                else:
                    p = inv_link(t[k] - eta) - inv_link(t[k - 1] - eta)

                # Avoid log(0)
                p = np.clip(p, 1e-10, 1)
                loss -= np.log(p)

        # L1 and L2 penalties
        if self.alpha_l1 > 0:
            loss += self.alpha_l1 * (np.sum(np.abs(A)) + np.sum(np.abs(B)))
            if self.random_slopes and Ag is not None:
                loss += self.alpha_l1 * np.sum(np.abs(Ag))
            if self.random_intercepts and Bg is not None:
                loss += self.alpha_l1 * np.sum(np.abs(Bg))

        if self.alpha_l2 > 0:
            loss += self.alpha_l2 * (np.sum(A**2) + np.sum(B**2))
            if self.random_slopes and Ag is not None:
                loss += self.alpha_l2 * np.sum(Ag**2)
            if self.random_intercepts and Bg is not None:
                loss += self.alpha_l2 * np.sum(Bg**2)

        return loss

    def fit(self, X, Y, groups):
        X = np.asarray(X)
        Y = np.asarray(Y)
        groups = np.asarray(groups)

        self.n_samples, self.n_features = X.shape
        _, self.n_outputs = Y.shape
        self.groups_, group_idx = np.unique(groups, return_inverse=True)
        self.n_groups = len(self.groups_)

        # Init
        rng = np.random.default_rng(42)
        A = rng.normal(0, 0.1, (self.n_outputs, self.n_features))
        B = rng.normal(0, 0.1, self.n_outputs)
        Ag = rng.normal(0, 0.01, (self.n_groups, self.n_outputs, self.n_features)) if self.random_slopes else None
        Bg = rng.normal(0, 0.01, (self.n_groups, self.n_outputs)) if self.random_intercepts else None
        thresholds = [np.linspace(-1, 1, nc - 1) for nc in self.n_cats]

        init_params = self._pack_params(A, B, Ag, Bg, thresholds)

        result = minimize(
            self._negative_log_likelihood,
            init_params,
            args=(X, Y, group_idx),
            method="L-BFGS-B",
            options={"maxiter": 500}
        )

        self.A_, self.B_, self.Ag_, self.Bg_, self.thresholds_ = self._unpack_params(result.x)
        self.group_idx_ = {g: i for i, g in enumerate(self.groups_)}
        return self

    def predict_proba(self, X, groups):
        X = np.asarray(X)
        groups = np.asarray(groups)

        proba = []
        for i in range(X.shape[0]):
            g = self.group_idx_[groups[i]]
            x_i = X[i]
            row_probs = []
            for j in range(self.n_outputs):
                eta = self.A_[j] @ x_i + self.B_[j]
                if self.random_slopes:
                    eta += self.Ag_[g, j] @ x_i
                if self.random_intercepts:
                    eta += self.Bg_[g, j]

                t = self.thresholds_[j]
                inv_link = self.family.inv_link
                k_cats = len(t) + 1
                probs = np.zeros(k_cats)
                for k in range(k_cats):
                    lo = -np.inf if k == 0 else t[k - 1]
                    hi = np.inf if k == k_cats - 1 else t[k]
                    probs[k] = inv_link(hi - eta) - inv_link(lo - eta)
                probs /= probs.sum()
                row_probs.append(probs)
            proba.append(row_probs)
        return proba

    def predict(self, X, groups):
        proba = self.predict_proba(X, groups)
        n_samples = X.shape[0]
        preds = np.zeros((n_samples, self.n_outputs), dtype=int)
        for i in range(n_samples):
            for j in range(self.n_outputs):
                preds[i, j] = np.argmax(proba[i][j])
        return preds
