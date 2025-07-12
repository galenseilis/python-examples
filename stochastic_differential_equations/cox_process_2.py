import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson

# ----------------- Simulation Parameters ----------------- #
T = 10.0  # Total time
dt = 0.01  # Time step
nt = int(T / dt)  # Number of time steps
time = np.linspace(0, T, nt + 1)

n = 3  # Dimension of latent state

# True parameters (for simulation)
A_true = np.array([[-0.1, 0.0, 0.05], [0.0, -0.2, 0.1], [0.0, 0.0, -0.1]])

theta_true = np.array([1.0, -1.0, 0.5])  # Intensity weight vector
sigma_true = 0.1  # Noise strength

x0 = np.array([1.0, 0.5, -0.2])  # Initial state

# ----------------- Simulate Latent SDE ----------------- #
X = np.zeros((nt + 1, n))
X[0] = x0

for t in range(nt):
    dW = np.random.normal(0, np.sqrt(dt), size=n)
    drift = A_true @ X[t]
    diffusion = sigma_true * dW
    X[t + 1] = X[t] + drift * dt + diffusion

# ----------------- Compute Intensity and Generate Counts ----------------- #
lambda_t = np.exp(X @ theta_true)  # Poisson intensity
counts = np.random.poisson(lambda_t * dt)  # Time series count data


# ----------------- Model Fitting (MLE) ----------------- #
def simulate_X(A_flat, sigma, x0):
    A = A_flat.reshape((n, n))
    X_sim = np.zeros_like(X)
    X_sim[0] = x0
    for t in range(nt):
        dW = np.random.normal(0, np.sqrt(dt), size=n)
        drift = A @ X_sim[t]
        diffusion = sigma * dW
        X_sim[t + 1] = X_sim[t] + drift * dt + diffusion
    return X_sim


def neg_log_likelihood(params):
    A_flat = params[: n * n]
    theta = params[n * n : n * n + n]
    sigma = np.exp(params[-1])  # ensure positivity
    X_sim = simulate_X(A_flat, sigma, x0)
    lambda_sim = np.exp(X_sim @ theta)
    expected_counts = lambda_sim * dt
    log_lik = poisson.logpmf(counts, expected_counts)
    return -np.sum(log_lik)


# Initial guess for optimization
init_params = np.concatenate([A_true.flatten(), theta_true, [np.log(sigma_true)]])
result = minimize(neg_log_likelihood, init_params, method="L-BFGS-B")

# Extract fitted parameters
A_fit = result.x[: n * n].reshape((n, n))
theta_fit = result.x[n * n : n * n + n]
sigma_fit = np.exp(result.x[-1])

# ----------------- Plotting ----------------- #
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Latent state trajectories
axs[0].plot(time, X, label=[f"x{i}" for i in range(n)])
axs[0].set_title("Latent State Trajectories (Simulated)")
axs[0].legend()
axs[0].grid(True)

# Intensity function
axs[1].plot(time, lambda_t, label="True Intensity λ(t)", color="darkred")
axs[1].set_title("Poisson Intensity Over Time")
axs[1].set_ylabel("λ(t)")
axs[1].grid(True)

# Observed count data
axs[2].plot(time, counts, label="Observed Counts", drawstyle="steps-post")
axs[2].set_title("Observed Count Time Series")
axs[2].set_xlabel("Time")
axs[2].set_ylabel("Counts")
axs[2].grid(True)

plt.tight_layout()
plt.show()

# Print results
print("Estimated A matrix:\n", A_fit)
print("Estimated theta:\n", theta_fit)
print("Estimated sigma:\n", sigma_fit)
