import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize
from collections import defaultdict

### LINK FUNCTIONS

class IdentityLink:
    def __call__(self, x):
        return x

    def inv(self, x):
        return x

class LogLink:
    def __call__(self, x):
        return np.log(np.clip(x, 1e-8, None))

    def inv(self, x):
        return np.exp(x)

class LogitLink:
    def __call__(self, x):
        eps = 1e-8
        return np.log(np.clip(x, eps, 1 - eps) / (1 - np.clip(x, eps, 1 - eps)))

    def inv(self, x):
        return 1 / (1 + np.exp(-x))

### FAMILY CLASSES

class NormalFamily:
    def __init__(self, link=None):
        self.link = link or IdentityLink()

    def log_likelihood(self, y, eta):
        mu = self.link.inv(eta)
        return -0.5 * np.sum((y - mu)**2)

class GammaFamily:
    def __init__(self, link=None, shape=2.0):
        self.link = link or LogLink()
        self.shape = shape

    def log_likelihood(self, y, eta):
        mu = self.link.inv(eta)
        return np.sum(
            self.shape * np.log(self.shape) + (self.shape - 1) * np.log(y) -
            self.shape * np.log(mu) - self.shape * y / mu - np.log(np.math.gamma(self.shape))
        )

class NegativeBinomialFamily:
    def __init__(self, link=None, n=2.0):
        self.link = link or LogLink()
        self.n = n

    def log_likelihood(self, y, eta):
        mu = self.link.inv(eta)
        p = self.n / (self.n + mu)
        return np.sum(
            np.log(np.math.gamma(y + self.n)) - np.log(np.math.gamma(self.n)) -
            np.log(np.math.factorial(y)) + self.n * np.log(p) + y * np.log(1 - p)
        )

class OrdinalFamily:
    def __init__(self, n_categories=3, thresholds=None, link=None):
        self.n_categories = n_categories
        self.link = link or LogitLink()
        if thresholds is None:
            self.thresholds = np.linspace(-1, 1, n_categories - 1)
        else:
            self.thresholds = np.array(thresholds)

    def log_likelihood(self, y, eta):
        thresholds = self.thresholds
        ll = 0.0
        for i in range(len(y)):
            k = int(y[i])
            if k == 0:
                p = 1 / (1 + np.exp(-(thresholds[0] - eta[i])))
            elif k == len(thresholds):
                p = 1 - 1 / (1 + np.exp(-(thresholds[-1] - eta[i])))
            else:
                upper = 1 / (1 + np.exp(-(thresholds[k] - eta[i])))
                lower = 1 / (1 + np.exp(-(thresholds[k - 1] - eta[i])))
                p = upper - lower
            ll += np.log(np.clip(p, 1e-10, None))
        return ll

### MODEL CLASS

class GNSEM:
    def __init__(self, structure, families):
        self.structure = structure  # dict: var_name -> list of parent var names
        self.families = families    # dict: var_name -> family instance
        self.coef_ = {}
        self.intercepts_ = {}

    def _pack_params(self, coefs, intercepts):
        return np.concatenate([np.concatenate([coefs[k].ravel(), [intercepts[k]]]) for k in self.structure])

    def _unpack_params(self, flat_params):
        coefs = {}
        intercepts = {}
        i = 0
        for var in self.structure:
            n_features = self.X_shapes_[var][1]
            coefs[var] = flat_params[i:i + n_features]
            i += n_features
            intercepts[var] = flat_params[i]
            i += 1
        return coefs, intercepts

    def fit(self, X_dict, Y_dict, groups=None):
        self.X_shapes_ = {k: X.shape for k, X in X_dict.items()}

        def objective(flat_params):
            coefs, intercepts = self._unpack_params(flat_params)
            loss = 0.0
            for var, parents in self.structure.items():
                X = X_dict[var]
                y = Y_dict[var]
                eta = X @ coefs[var] + intercepts[var]
                family = self.families[var]
                loss -= family.log_likelihood(y, eta)
            return loss

        grad_obj = grad(objective)

        # Initial parameters
        init_params = []
        for var in self.structure:
            X = X_dict[var]
            init_params.append(np.zeros(X.shape[1]))  # coefs
            init_params.append(np.array([0.0]))        # intercept
        flat_init = np.concatenate(init_params)

        res = minimize(objective, flat_init, jac=grad_obj, method='L-BFGS-B')
        coefs, intercepts = self._unpack_params(res.x)
        self.coef_ = coefs
        self.intercepts_ = intercepts

    def predict(self, X_dict, groups=None):
        preds = {}
        for var in self.structure:
            X = X_dict[var]
            coef = self.coef_[var]
            intercept = self.intercepts_[var]
            eta = X @ coef + intercept
            preds[var] = self.families[var].link.inv(eta)
        return preds

# Exports
__all__ = [
    'GNSEM',
    'NormalFamily',
    'GammaFamily',
    'NegativeBinomialFamily',
    'OrdinalFamily',
    'IdentityLink',
    'LogLink',
    'LogitLink',
]
