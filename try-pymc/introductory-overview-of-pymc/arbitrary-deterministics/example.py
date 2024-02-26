import arviz as az
import matplotlib.pyplot  as plt
from pytensor.compile.ops import as_op
import pytensor.tensor as at
import pymc as pm

@as_op(itypes=[at.lscalar], otypes=[at.lscalar])
def crazy_modulo3(value):
	if value > 0:
		return value % 3
	return (- value + 1) % 3

with pm.Model() as model_deterministic:
	a = pm.Poisson('a', 1)
	b = pm.Deterministic('b', crazy_modulo3(a))

with model_deterministic:
	prior_samples = pm.sample_prior_predictive(10_000)

az.plot_dist(
	prior_samples.prior['b'],
	kind='hist',
	label='simulated'
	)

plt.tight_layout()
plt.savefig('crazy_modulo_3_prior_predictive_hist.png', dpi=300)
plt.close()
