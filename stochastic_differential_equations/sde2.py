from collections.abc import Callable
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# --- SDE: dX_t = mu * X_t * dt + sigma * X_t * dW_t ---
mu_val: float = 0.1
sigma_val: float = 0.3


def mu(x: float, t: float) -> float:
    return mu_val * x


def sigma(x: float, t:float) -> float:
    return sigma_val * x


def sigma_prime(x: float, t: float) -> float:
    """Derivative of sigma w.r.t x (used in Milstein and Platen)"""
    return sigma_val  # since sigma(x, t) = 0.3 * x, the derivative w.r.t x is 0.3


# --- Euler–Maruyama (strong order 0.5, weak order 1.0) ---
def euler_maruyama(mu: Callable, sigma: Callable, x0: float, T: float, N: int) -> NDArray[np.float64]:
    dt: float = T / N
    x: NDArray[np.float64] = np.zeros(N + 1)
    x[0] = x0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt))
        x[i + 1] = x[i] + mu(x[i], i * dt) * dt + sigma(x[i], i * dt) * dW
    return x


# --- Milstein Method (strong order 1.0, weak order 1.0) ---
def milstein(mu, sigma, sigma_prime, x0, T, N):
    dt = T / N
    x = np.zeros(N + 1)
    x[0] = x0
    for i in range(N):
        dW = np.random.normal(0.0, np.sqrt(dt))
        x[i + 1] = (
            x[i]
            + mu(x[i], i * dt) * dt
            + sigma(x[i], i * dt) * dW
            + 0.5 * sigma(x[i], i * dt) * sigma_prime(x[i], i * dt) * (dW**2 - dt)
        )
    return x


# --- Heun (Stochastic RK) (strong order 1.0, weak order 1.0) ---
def heun(mu, sigma, x0, T, N):
    dt = T / N
    x = np.zeros(N + 1)
    x[0] = x0
    for i in range(N):
        t = i * dt
        dW = np.random.normal(0.0, np.sqrt(dt))
        drift1 = mu(x[i], t)
        diff1 = sigma(x[i], t)
        x_tilde = x[i] + drift1 * dt + diff1 * dW
        drift2 = mu(x_tilde, t + dt)
        diff2 = sigma(x_tilde, t + dt)
        x[i + 1] = x[i] + 0.5 * (drift1 + drift2) * dt + 0.5 * (diff1 + diff2) * dW
    return x


# --- Platen's Method (SRK order 1.5 strong, 2.0 weak) ---
def platen(mu, sigma, sigma_prime, x0, T, N):
    dt = T / N
    x = np.zeros(N + 1)
    x[0] = x0
    for i in range(N):
        t = i * dt
        dW = np.random.normal(0.0, np.sqrt(dt))
        dZ = (dW**2 - dt) / np.sqrt(2 * dt)

        f = mu(x[i], t)
        g = sigma(x[i], t)
        g_prime = sigma_prime(x[i], t)

        x[i + 1] = (
            x[i]
            + f * dt
            + g * dW
            + 0.5 * g * g_prime * (dW**2 - dt)
            + (f * g_prime - 0.5 * g**2 * g_prime**2) * dZ * dt
        )
    return x


# --- Simulation Parameters ---
x0 = 1.0
T = 1.0
N = 500
plt.figure(figsize=(10, 6))
for seed in range(100):
    time_grid = np.linspace(0, T, N + 1)

    # Use same seed for fair comparison
    np.random.seed(seed)

    x_euler = euler_maruyama(mu, sigma, x0, T, N)

    np.random.seed(seed)
    x_milstein = milstein(mu, sigma, sigma_prime, x0, T, N)

    # np.random.seed(seed)
    # x_heun = heun(mu, sigma, x0, T, N)
    #
    # np.random.seed(seed)
    # x_platen = platen(mu, sigma, sigma_prime, x0, T, N)

    # --- Plotting ---
    if not seed:
        plt.plot(
            time_grid,
            x_euler,
            "--",
            label="Euler–Maruyama (0.5)",
            alpha=0.7,
            color="red",
        )
        plt.plot(time_grid, x_milstein, label="Milstein (1.0)", alpha=0.9, color="blue")
        # plt.plot(time_grid, x_heun, label='Heun RK (1.0)', alpha=0.9, color='green')
        # plt.plot(time_grid, x_platen, label='Platen SRK (1.5)', linewidth=2, color='magenta')
    else:
        plt.plot(time_grid, x_euler, "--", alpha=0.7, color="red")
        plt.plot(time_grid, x_milstein, alpha=0.9, color="blue")
        # plt.plot(time_grid, x_heun,  alpha=0.9, color='green')
        # plt.plot(time_grid, x_platen,  linewidth=2, color='magenta')

plt.title("Comparison of SDE Solvers (Geometric Brownian Motion)")
plt.xlabel("Time t")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
