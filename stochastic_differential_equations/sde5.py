import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Dimension
n = 3
T = 1.0
N = 100
dt = T / N
time_grid = np.linspace(0, T, N + 1)

# --- True parameters ---
Q_true = np.array([[-0.3, 0.2, 0.1], [0.1, -0.4, 0.3], [0.2, 0.2, -0.4]])

B_true = np.array([[0.3, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.2]])

X0 = np.array([1.0, 0.0, 0.0])


def mu_factory(Q):
    return lambda X, t: Q @ X


def sigma(X, t):
    return B_true @ X


# Euler-Maruyama solver
def euler_maruyama(mu, sigma, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        X[i + 1] = X[i] + mu(X[i], i * dt) * dt + sigma(X[i], i * dt) * dW
    return X


# --- Generate synthetic "observed" data ---
np.random.seed(123)
mu_true = mu_factory(Q_true)
X_data = euler_maruyama(mu_true, sigma, X0, T, N)

# Add some observational noise (e.g., Gaussian noise)
obs_noise_std = 0.05
X_obs = X_data + np.random.normal(0, obs_noise_std, X_data.shape)

# --- Parameter vectorization ---
# Flatten off-diagonal elements of Q as free parameters,
# diagonal constrained as negative row sums to ensure generator property


def Q_from_params(params):
    Q = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = max(params[idx], 0)  # constrain off-diagonal >= 0
                idx += 1
    # Diagonal entries ensure row sums zero
    for i in range(n):
        Q[i, i] = -np.sum(Q[i, :])
    return Q


# --- Loss function ---
def loss(params):
    Q_est = Q_from_params(params)
    mu_est = mu_factory(Q_est)
    np.random.seed(0)  # fix seed for reproducibility of simulation inside optimizer
    X_sim = euler_maruyama(mu_est, sigma, X0, T, N)
    # Mean squared error between simulated and observed paths
    return np.mean((X_sim - X_obs) ** 2)


# Initial guess (all off-diagonals 0.1)
init_params = np.full(n * (n - 1), 0.1)

# --- Run optimization ---
result = minimize(
    loss, init_params, method="L-BFGS-B", bounds=[(0, None)] * len(init_params)
)

print("Optimization success:", result.success)
print("Estimated off-diagonal Q params:", result.x)

Q_est = Q_from_params(result.x)
print("Estimated Q matrix:\n", Q_est)

# --- Simulate with estimated Q ---
mu_est = mu_factory(Q_est)
np.random.seed(123)
X_fit = euler_maruyama(mu_est, sigma, X0, T, N)

# --- Plot results ---
plt.figure(figsize=(10, 6))
labels = ["X₁", "X₂", "X₃"]
for i in range(n):
    plt.plot(
        time_grid,
        X_obs[:, i],
        "o",
        markersize=3,
        label=f"Observed {labels[i]}",
        alpha=0.5,
    )
    plt.plot(time_grid, X_fit[:, i], "-", label=f"Fitted {labels[i]}")
plt.xlabel("Time")
plt.ylabel("State")
plt.title("Fitting drift matrix Q to observed data")
plt.legend()
plt.show()
