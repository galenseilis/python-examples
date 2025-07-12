import numpy as np
import matplotlib.pyplot as plt

# --- Dimension ---
n = 3

# --- Generator matrix Q: rate transition matrix ---
# Rows sum to zero, off-diagonal >= 0
Q = np.array([[-0.3, 0.2, 0.1], [0.1, -0.4, 0.3], [0.2, 0.2, -0.4]])

# --- Diffusion matrix B ---
B = np.array([[0.3, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.2]])


# --- Drift function: mu(X, t) = Q X ---
def mu(X, t):
    return Q @ X


# --- Diffusion function: sigma(X, t) = B X ---
def sigma(X, t):
    return B @ X


# --- Diffusion derivative ---
# For linear sigma, sigma_prime = B (constant)
def sigma_prime(X, t):
    return B


# --- Euler–Maruyama method ---
def euler_maruyama(mu, sigma, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(
            0.0, np.sqrt(dt), size=len(X0)
        )  # vector Brownian increment
        X[i + 1] = X[i] + mu(X[i], i * dt) * dt + sigma(X[i], i * dt) * dW
    return X


# --- Milstein method ---
def milstein(mu, sigma, sigma_prime, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        # Elementwise multiplication for Milstein correction
        correction = (
            0.5 * sigma(X[i], i * dt) * (sigma_prime(X[i], i * dt) @ (dW**2 - dt))
        )
        X[i + 1] = X[i] + mu(X[i], i * dt) * dt + sigma(X[i], i * dt) * dW + correction
    return X


# --- Heun method (predictor-corrector) ---
def heun(mu, sigma, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        t = i * dt
        drift1 = mu(X[i], t)
        diff1 = sigma(X[i], t)
        X_tilde = X[i] + drift1 * dt + diff1 * dW
        drift2 = mu(X_tilde, t + dt)
        diff2 = sigma(X_tilde, t + dt)
        X[i + 1] = X[i] + 0.5 * (drift1 + drift2) * dt + 0.5 * (diff1 + diff2) * dW
    return X


# --- Platen method (strong order 1.5) ---
def platen(mu, sigma, sigma_prime, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        dZ = (dW**2 - dt) / np.sqrt(2 * dt)
        t = i * dt
        f = mu(X[i], t)
        g = sigma(X[i], t)
        g_prime = sigma_prime(X[i], t)
        # Note: @ is matrix multiplication, * elementwise
        term1 = f * dt
        term2 = g * dW
        term3 = 0.5 * g * (g_prime @ (dW**2 - dt))
        term4 = (f * np.diag(g_prime) - 0.5 * g**2 * np.diag(g_prime) ** 2) * dZ * dt
        X[i + 1] = X[i] + term1 + term2 + term3 + term4
    return X


# --- Simulation parameters ---
T = 1.0
N = 500
X0 = np.array([1.0, 0.0, 0.0])  # start mostly in state 1
time_grid = np.linspace(0, T, N + 1)

# Fix random seed for reproducibility
np.random.seed(42)
X_euler = euler_maruyama(mu, sigma, X0, T, N)

np.random.seed(42)
X_milstein = milstein(mu, sigma, sigma_prime, X0, T, N)

np.random.seed(42)
X_heun = heun(mu, sigma, X0, T, N)

np.random.seed(42)
X_platen = platen(mu, sigma, sigma_prime, X0, T, N)

# --- Plot results ---
plt.figure(figsize=(12, 6))
labels = ["X₁(t)", "X₂(t)", "X₃(t)"]
colors = ["tab:blue", "tab:orange", "tab:green"]

for dim in range(n):
    plt.plot(
        time_grid, X_platen[:, dim], label=f"Platen - {labels[dim]}", color=colors[dim]
    )
    plt.plot(
        time_grid,
        X_heun[:, dim],
        "--",
        label=f"Heun - {labels[dim]}",
        color=colors[dim],
        alpha=0.7,
    )
    plt.plot(
        time_grid,
        X_milstein[:, dim],
        ":",
        label=f"Milstein - {labels[dim]}",
        color=colors[dim],
        alpha=0.7,
    )
    plt.plot(
        time_grid,
        X_euler[:, dim],
        "--",
        label=f"Euler - {labels[dim]}",
        color=colors[dim],
        alpha=0.3,
    )

plt.title("SDE with Generator Matrix Drift (Rate Transition Matrix) and Diffusion")
plt.xlabel("Time t")
plt.ylabel("X(t)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
