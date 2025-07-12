import numpy as np
import matplotlib.pyplot as plt

# --- Matrix SDE: dX = A X dt + B X dW ---
n = 3  # dimension of system

A = np.array([[0.1, 0.0, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, -0.1]])

B = np.array([[0.3, 0.0, 0.0], [0.0, 0.4, 0.0], [0.0, 0.0, 0.2]])


def mu(X, t):
    return A @ X  # drift: linear


def sigma(X, t):
    return B @ X  # diffusion: linear


def sigma_prime(X, t):
    # Jacobian approximation (diagonal case)
    return B  # constant in this linear example


# --- Vector Euler–Maruyama ---
def euler_maruyama(mu, sigma, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        X[i + 1] = X[i] + mu(X[i], i * dt) * dt + sigma(X[i], i * dt) * dW
    return X


# --- Vector Milstein ---
def milstein(mu, sigma, sigma_prime, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        X[i + 1] = (
            X[i]
            + mu(X[i], i * dt) * dt
            + sigma(X[i], i * dt) * dW
            + 0.5 * (B @ X[i]) * (B @ np.ones_like(X[i])) * (dW**2 - dt)
        )
    return X


# --- Vector Heun (Stochastic RK) ---
def heun(mu, sigma, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        drift1 = mu(X[i], i * dt)
        diff1 = sigma(X[i], i * dt)
        X_tilde = X[i] + drift1 * dt + diff1 * dW
        drift2 = mu(X_tilde, (i + 1) * dt)
        diff2 = sigma(X_tilde, (i + 1) * dt)
        X[i + 1] = X[i] + 0.5 * (drift1 + drift2) * dt + 0.5 * (diff1 + diff2) * dW
    return X


# --- Vector Platen (simplified form) ---
def platen(mu, sigma, sigma_prime, X0, T, N):
    dt = T / N
    X = np.zeros((N + 1, len(X0)))
    X[0] = X0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt), size=len(X0))
        dZ = (dW**2 - dt) / np.sqrt(2 * dt)
        f = mu(X[i], i * dt)
        g = sigma(X[i], i * dt)
        g_prime = sigma_prime(X[i], i * dt)
        X[i + 1] = (
            X[i]
            + f * dt
            + g * dW
            + 0.5 * g * g_prime @ (dW**2 - dt)
            + (f * np.diag(g_prime) - 0.5 * g**2 * np.diag(g_prime) ** 2) * dZ * dt
        )
    return X


# --- Simulation ---
T = 1.0
N = 500
X0 = np.array([1.0, 2.0, 3.0])
time_grid = np.linspace(0, T, N + 1)

np.random.seed(42)
X_euler = euler_maruyama(mu, sigma, X0, T, N)

np.random.seed(42)
X_milstein = milstein(mu, sigma, sigma_prime, X0, T, N)

np.random.seed(42)
X_heun = heun(mu, sigma, X0, T, N)

np.random.seed(42)
X_platen = platen(mu, sigma, sigma_prime, X0, T, N)

# --- Plot Each Component ---
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

plt.title("Vector-Valued SDE Simulation: dX = A X dt + B X dW")
plt.xlabel("Time t")
plt.ylabel("X(t)")
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
plt.grid(True)
plt.tight_layout()
plt.show()
