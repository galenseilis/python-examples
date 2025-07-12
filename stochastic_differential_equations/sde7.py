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

B_true = np.diag([0.3, 0.4, 0.2])  # Diagonal diffusion matrix

X0 = np.array([1.0, 0.0, 0.0])


def mu_factory(Q):
    return lambda X, t: Q @ X


def sigma_factory(B):
    return lambda X, t: B @ X


# Euler-Maruyama solver
def euler_maruyama(mu, sigma, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        X[i + 1] = X[i] + mu(X[i], i * dt) * dt + sigma(X[i], i * dt) * dW
    return X


# --- Generate synthetic observed data ---
np.random.seed(123)
mu_true = mu_factory(Q_true)
sigma_true = sigma_factory(B_true)
X_data = euler_maruyama(mu_true, sigma_true, X0, T, N)

# Add observational noise
obs_noise_std = 0.05
X_obs = X_data + np.random.normal(0, obs_noise_std, X_data.shape)

# --- Parameter vectorization ---

# Number of off-diagonal Q params
num_q_params = n * (n - 1)
# Number of B params (diagonal elements)
num_b_params = n


def Q_from_params(params_q):
    Q = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                Q[i, j] = max(params_q[idx], 0)  # enforce >=0
                idx += 1
    for i in range(n):
        Q[i, i] = -np.sum(Q[i, :])
    return Q


def B_from_params(params_b):
    return np.diag(np.maximum(params_b, 0))  # diagonal positive


# --- Loss function ---
def loss(params):
    params_q = params[:num_q_params]
    params_b = params[num_q_params:]
    Q_est = Q_from_params(params_q)
    B_est = B_from_params(params_b)
    mu_est = mu_factory(Q_est)
    sigma_est = sigma_factory(B_est)
    np.random.seed(0)  # fixed seed for reproducibility
    X_sim = euler_maruyama(mu_est, sigma_est, X0, T, N)
    return np.mean((X_sim - X_obs) ** 2)


# Initial guess: off-diagonal Q = 0.1, B diagonal = 0.1
init_q = np.full(num_q_params, 0.1)
init_b = np.full(num_b_params, 0.1)
init_params = np.concatenate([init_q, init_b])

# Bounds: Q off-diagonal >=0, B diagonal >=0
bounds = [(0, None)] * (num_q_params + num_b_params)

# --- Run optimization ---
result = minimize(loss, init_params, method="L-BFGS-B", bounds=bounds)

print("Optimization success:", result.success)

Q_est = Q_from_params(result.x[:num_q_params])
B_est = B_from_params(result.x[num_q_params:])

print("Estimated Q matrix:\n", Q_est)
print("Estimated B matrix (diagonal):\n", B_est)

# --- Simulate with estimated parameters ---
mu_est = mu_factory(Q_est)
sigma_est = sigma_factory(B_est)
np.random.seed(123)
X_fit = euler_maruyama(mu_est, sigma_est, X0, T, N)

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
plt.title("Fitting drift matrix Q and diffusion matrix B to observed data")
plt.legend()
plt.show()
