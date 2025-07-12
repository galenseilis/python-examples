from collections.abc import Callable
import numpy as np
from numpy._typing import _nested_sequence
from numpy.typing import NDArray
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# Define the SDE: dX_t = mu(X_t, t) dt + sigma(X_t, t) dW_t
# This example uses Geometric Brownian Motion:
# μ(x, t) = 0.1 * x      (drift)
# σ(x, t) = 0.3 * x      (diffusion)
# ----------------------------------------------------------


def mu(x: float, t: float) -> float | np.float64:
    """Drift function μ(x, t)

    Args:
        x (float): State coordinate.
        t (float): Time coordinate.

    Returns:
        float: Drift term.
    """
    return 0.1 * x


def sigma(x: float, t: float) -> float | np.float64:
    """Diffusion function σ(x, t)

    Args:
        x (float): State coordinate.
        t (float): Time coordinate.

    Returns:
        float: Diffusion term.
    """
    return 0.3 * x


# ------------------------------------------
# Euler-Maruyama method for SDE
# ------------------------------------------
def euler_maruyama(
    mu: Callable[[float, float], float],
    sigma: Callable[[float, float], float],
    x0: float,
    t0: float,
    T: float,
    N: int
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:

    dt: float = (T - t0) / N
    t: NDArray[np.float64] = np.linspace(t0, T, N + 1)
    x : NDArray[np.float64] = np.zeros(N + 1)
    x[0] = x0

    for i in range(N):
        t_i = t[i]
        x_i = x[i]
        dW = np.random.normal(0.0, np.sqrt(dt))  # Brownian increment ΔW
        x[i + 1] = x_i + mu(x_i, t_i) * dt + sigma(x_i, t_i) * dW

    return t, x


# ----------------------------------------------------
# Runge-Kutta (Heun) method for SDE (strong order 1.0)
# ----------------------------------------------------
def heun_runge_kutta(mu: Callable, sigma: Callable, x0: float, t0: float, T: float, N: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    dt: float = (T - t0) / N
    t: NDArray[np.float64] = np.linspace(t0, T, N + 1)
    x: NDArray[np.float64]= np.zeros(N + 1)
    x[0] = x0

    for i in range(N):
        t_i = t[i]
        x_i = x[i]
        dW = np.random.normal(0.0, np.sqrt(dt))  # Brownian increment ΔW

        # First evaluations at (x_i, t_i)
        drift1 = mu(x_i, t_i)
        diff1 = sigma(x_i, t_i)

        # Predictor (Euler step)
        x_tilde = x_i + drift1 * dt + diff1 * dW

        # Evaluate at predicted point (x_tilde, t_{i+1})
        drift2 = mu(x_tilde, t_i + dt)
        diff2 = sigma(x_tilde, t_i + dt)

        # Corrector (Heun update)
        x[i + 1] = x_i + 0.5 * (drift1 + drift2) * dt + 0.5 * (diff1 + diff2) * dW

    return t, x


# ------------------------------------------
# Parameters for simulation
# ------------------------------------------
x0 = 1.0  # Initial condition X₀
t0 = 0.0  # Start time
T = 1.0  # End time
N = 1000  # Number of time steps

# Simulate one path with both methods
# np.random.seed(42)  # For reproducibility
t_euler, x_euler = euler_maruyama(mu, sigma, x0, t0, T, N)
# np.random.seed(42)  # Reset seed so both methods use same Brownian path
t_heun, x_heun = heun_runge_kutta(mu, sigma, x0, t0, T, N)

# ------------------------------------------
# Plotting
# ------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(t_euler, x_euler, label="Euler-Maruyama", linestyle="--", alpha=0.7)
plt.plot(t_heun, x_heun, label="Heun Runge-Kutta (strong order 1.0)", linewidth=2)
plt.title("Solving an SDE with Euler–Maruyama vs Heun Runge–Kutta")
plt.xlabel("Time t")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
