import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as at

# Load data
test_scores = pd.read_csv(
	pm.get_data('test_scores.csv'),
	index_col=0
)

# Peek at data
(
	test_scores
	.head()
	.to_csv('example_data_head.csv', index=False)
)

# Plot histogram of scores
test_scores['score'].hist()
plt.savefig('score_histogram.png', dpi=300)
plt.close()

# Process data

## Dropping missing values; done here only for simplicity.
X = test_scores.dropna().astype(float)
y = X.pop('score')

## Standardize the features
X -= X.mean()
X /= X.std()

N, D = X.shape

# Model Specification

## Intial guess for D0 in Half-StudentT2 dist
D0 = int(D / 2)

with pm.Model(coords={"predictors": X.columns.values}) as test_score_model:
	# Prior on error SD
	sigma = pm.HalfNormal("sigma", 25)

	# Global shrinkage prior
	tau = pm.HalfStudentT("tau", 2, D0 / (D - D0) * sigma / np.sqrt(N))

	# Local shrinkage prior
	lam = pm.HalfStudentT("lam", 5, dims='predictors')
	c2 = pm.InverseGamma("c2", 1, 1)
	z = pm.Normal("z", 0.0, 1.0, dims='predictors')

	# Shrunk coefficients
	beta = pm.Deterministic(
		"beta",
		z * tau * lam * at.sqrt(c2 / (c2 + tau ** 2 + lam ** 2)), 
		dims='predictors'
	)

	# No shrinkage on intercept
	beta0 = pm.Normal("beta0", 100, 25.0)

	scores = pm.Normal(
		"scores",
		beta0 + at.dot(X.values, beta),
		sigma,
		observed=y.values
		)

# Visualize model
model_dot = pm.model_to_graphviz(test_score_model)
model_dot.render('test_score_model')

# Prior predictive sample from the model
with test_score_model:
	prior_samples = pm.sample_prior_predictive(100)

# Plot prior predictive against sample
az.plot_dist(
	test_scores['score'].values,
	kind='hist',
	color='C1',
	hist_kwargs=dict(alpha=0.6),
	label="observed"
	)

az.plot_dist(
	prior_samples.prior_predictive['scores'],
	kind='hist',
	hist_kwargs=dict(alpha=0.6),
	label='simulated'
	)

plt.xticks(rotation=45)
plt.savefig('prior_predictive_dist_check.png', dpi=300)
plt.close()

# Model fitting
with test_score_model:
	idata = pm.sample(1000, tune=2000, random_seed=2018, target_accept=0.99)

# Plot Trace
az.plot_trace(idata, var_names=['tau', 'sigma', 'c2'])
plt.tight_layout()
plt.savefig('posterior_trace.png', dpi=300)
plt.close()

# Plot Energy
az.plot_energy(idata)
plt.savefig('sampling_energy.png', dpi=300)
plt.close()

# Plot forest of beta parameter
az.plot_forest(idata, var_names=['beta'], combined=True, hdi_prob=0.99, r_hat=True)
plt.tight_layout()
plt.savefig('forest_plot_beta_param.png', dpi=300)
plt.close()


