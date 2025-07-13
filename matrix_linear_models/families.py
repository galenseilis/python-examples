import numpy as np

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
