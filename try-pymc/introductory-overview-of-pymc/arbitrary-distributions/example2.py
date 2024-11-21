import pymc as pm
import pytensor.tensor as at


class BetaRV(at.random.op.RandomVariable):
    name = "beta"
    ndim_supp = 0
    ndims_params = []
    dtype = "floatX"

    @classmethod
    def rng_fn(cls, rng, size):
        raise NotImplementedError("Cannot sample from beta variable.")


beta = BetaRV()


class Beta(pm.Continuous):
    rv_op = beta

    @classmethod
    def dist(cls, mu=0, **kwargs):
        mu = at.as_tensor_variable(mu)
        return super().dist([mu], **kwargs)

    def logp(self, value):
        mu = self.mu
        return beta_logp(value - mu)


def beta_logp(value):
    return -1.5 * at.log(1 + (value) ** 2)


with pm.Model() as model:
    beta = Beta("beta", mu=0)
