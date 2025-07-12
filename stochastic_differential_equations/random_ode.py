import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Ground truth parameters for generating data
mu_true = 2.0
sigma_true = 0.3

# Domain
x = np.linspace(0, 1.0, 100)


# Analytical expressions for mean and variance
def y_mean(x, mu, sigma):
    return np.exp(-mu * x + 0.5 * sigma**2 * x**2)


def y_std(x, mu, sigma):
    return np.sqrt(
        np.exp(-2 * mu * x + 2 * sigma**2 * x**2) - y_mean(x, mu, sigma) ** 2
    )


# Generate observed data
observed_mean = y_mean(x, mu_true, sigma_true)
observed_std = y_std(x, mu_true, sigma_true)


# Loss function (fit both mean and std)
def loss(mu_sigma):
    mu, sigma = mu_sigma
    if sigma <= 0:
        return np.inf
    pred_mean = y_mean(x, mu, sigma)
    pred_std = y_std(x, mu, sigma)

    mean_loss = np.mean((pred_mean - observed_mean) ** 2)
    std_loss = np.mean((pred_std - observed_std) ** 2)
    return mean_loss + 0.5 * std_loss


# Fit parameters
initial_guess = [1.0, 0.5]
bounds = [(0.01, 5.0), (0.01, 2.0)]

result = minimize(loss, initial_guess, bounds=bounds, method="L-BFGS-B")
mu_fit, sigma_fit = result.x

print(f"True parameters: mu = {mu_true}, sigma = {sigma_true}")
print(f"Fitted parameters: mu = {mu_fit:.4f}, sigma = {sigma_fit:.4f}")

# Compute fitted curves
fitted_mean = y_mean(x, mu_fit, sigma_fit)
fitted_std = y_std(x, mu_fit, sigma_fit)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(x, observed_mean, label="Observed Mean", linewidth=2)
plt.fill_between(
    x,
    observed_mean - observed_std,
    observed_mean + observed_std,
    alpha=0.3,
    label="Observed ±1 std",
    color="gray",
)

plt.plot(x, fitted_mean, "--", label="Fitted Mean", linewidth=2)
plt.fill_between(
    x,
    fitted_mean - fitted_std,
    fitted_mean + fitted_std,
    alpha=0.3,
    label="Fitted ±1 std",
    color="orange",
)

plt.title("Analytical Fit of Distribution over ODE Parameter")
plt.xlabel("x")
plt.ylabel("y(x)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
