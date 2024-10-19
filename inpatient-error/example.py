import pymc as pm
import matplotlib.pyplot as plt
import numpy as np

# Generate some data from an exponential distribution
np.random.seed(123)
data = np.random.exponential(scale=1.0, size=100)

# Create PyMC model
with pm.Model() as model:
    # Exponential distribution parameter
    lambda_ = pm.Exponential('lambda_', lam=1.0)
    
    # Observed data
    obs = pm.Exponential('obs', lam=lambda_, observed=data)
    
    # Round the sampled exponential values
    rounded_obs = pm.Deterministic('rounded_obs', pm.math.ceil(obs))

# Sampling
with model:
    trace = pm.sample(1000, tune=1000)

# Plotting
pm.plot_trace(trace)

plt.show()
