import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc as pm

print(f"Running on PyMC v{pm.__version__}")

# CONFIG
az.style.use("arviz-darkgrid")

# Define data
y = np.array([28, 8, -3, 7, -1, 1, 18, 12])
sigma = np.array([15, 10, 16, 11, 9, 11, 10, 18])
J = len(y)

# Pooled model
with pm.Model() as pooled:
    # Latent pooled effect size

    mu = pm.Normal("mu", 0, sigma=1e6)
    obs = pm.Normal("obs", mu, sigma=sigma, observed=y)
    trace_p = pm.sample(2000)

# Plot trace of pooled model
az.plot_trace(trace_p)
plt.tight_layout()
plt.savefig("pooled_model_trace.png", dpi=300)
plt.close()


# Hierarchical Model

with pm.Model() as hierarchical:
    eta = pm.Normal("eta", 0, 1, shape=J)
    # Hierarchical mean and SD
    mu = pm.Normal("mu", 0, sigma=10)
    tau = pm.HalfNormal("tau", 10)

    # Non-centered parameterization of random effect
    theta = pm.Deterministic("theta", mu + tau * eta)

    obs = pm.Normal("obs", theta, sigma=sigma, observed=y)

    trace_h = pm.sample(2000, target_accept=0.9)

az.plot_trace(trace_h, var_names="mu")
plt.tight_layout()
plt.savefig("hierarchical_model_trace.png", dpi=300)
plt.close()

az.plot_forest(trace_h, var_names="theta")
plt.tight_layout()
plt.savefig("h_model_forest_theta.png", dpi=300)
plt.close()

# Model Log-Likelihood
with pooled:
    pm.compute_log_likelihood(trace_p)

with hierarchical:
    pm.compute_log_likelihood(trace_h)

df_comp_loo = az.compare({"hierarchical": trace_h, "pooled": trace_p})

az.plot_compare(df_comp_loo, insample_dev=False)
plt.tight_layout()
plt.savefig("model_comparison_plot.png", dpi=300)
plt.close()
