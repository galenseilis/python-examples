import arviz as az
import bambi as bmb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import xarray as xr

from pymc import HalfCauchy, Model, Normal, sample

print(f"Running on PyMC v{pm.__version__}")

# CONFIG
RANDOM_SEED = 2018
rng = np.random.default_rng(RANDOM_SEED)
az.style.use("arviz-darkgrid")

# Generating data
size = 200
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)

## y = a + b*x
true_regression_line = true_intercept + true_slope * x

## Add noise
y = true_regression_line + rng.normal(scale=0.5, size=size)

data = pd.DataFrame(dict(x=x, y=y))

# Plot Data
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(
	111,
	xlabel='x',
	ylabel='y',
	title='Generated Data'
	)
ax.plot(x, y, "x", label="sampled data")
ax.plot(x, true_regression_line, label="true regression line", lw=2.0)
plt.legend(loc=0)
plt.savefig('generated_data.png', dpi=300)
plt.close()

# Estimating the model
with Model() as model:  # model specifications in PyMC are wrapped in a with-statement
    # Define priors
    sigma = HalfCauchy("sigma", beta=10)
    intercept = Normal("Intercept", 0, sigma=20)
    slope = Normal("slope", 0, sigma=20)

    # Define likelihood
    likelihood = Normal("y", mu=intercept + slope * x, sigma=sigma, observed=y)

    # Inference!
    # draw 3000 posterior samples using NUTS sampling
    idata = sample(3000)

# Estimating with Bambi model
model = bmb.Model('y ~ x', data)
idata = model.fit(draws=10_000)

# Plot posterior trace
az.plot_trace(idata, figsize=(10,7))
plt.tight_layout()
plt.savefig('posterior_trace.png', dpi=300)
plt.close()

# Plot posterior predictive lines
idata.posterior["y_model"] = idata.posterior["Intercept"] + idata.posterior["x"] * xr.DataArray(x)
_, ax = plt.subplots(figsize=(7, 7))
az.plot_lm(idata=idata, y="y", num_samples=100, axes=ax, y_model="y_model")
ax.set_title("Posterior predictive regression lines")
ax.set_xlabel("x")
plt.tight_layout()
plt.savefig('posterior_predictive_lines.png', dpi=300)
plt.close()
