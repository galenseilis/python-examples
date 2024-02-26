import arviz as az
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
import numpy as np

RANDOM_SEED = 2018
rng = np.random.default_rng(RANDOM_SEED)
az.style.use('arviz-darkgrid')

# True parameter values
alpha, sigma = 1, 1
beta = [1, 2.5]

# Size of dataset
size = 100

# Predictor variable
X1 = np.random.randn(size)
X2 = np.random.randn(size) * 0.2

# Simulation outcome variable
Y = alpha + beta[0] * X1 + beta[1] * X2 + rng.normal(size=size) * sigma

# Plot data
fig, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 4))
axes[0].scatter(X1, Y, alpha=0.6)
axes[1].scatter(X2, Y, alpha=0.6)
axes[0].set_ylabel("Y")
axes[0].set_xlabel("X1")
axes[1].set_xlabel("X2")
plt.savefig('example1_data.png', dpi=300)
plt.close()

# Model Specification
print(F"Running PyMC v{pm.__version__}")

basic_model = pm.Model()
with basic_model:
	# Priors for unknown model parameters
	alpha = pm.Normal('alpha', mu=0, sigma=10)
	beta = pm.Normal('beta', mu=0, sigma=10, shape=2)
	sigma = pm.HalfNormal('sigma', sigma=1)

	# Expected value of the outcome
	mu = alpha + beta[0] * X1 + beta[1] * X2

	# Likelihood (sampling distribution) of observations
	y_obs = pm.Normal('Y_obs', mu=mu, sigma=sigma, observed=Y)

# Sampling from model
with basic_model:
	# Draw 1000 posterior samples
	idata = pm.sample()

with basic_model:
	# Instantiate sampler
	step = pm.Slice()

	# Draw 5000 posterior samples
	slice_data = pm.sample(5000, step=step)


# Posterior results
az.plot_trace(idata, combined=True)
plt.savefig('example1_traceplot.png', dpi=300)
plt.close()

az.summary(idata, round_to=2).to_csv('example1_posterior_summary.csv')
