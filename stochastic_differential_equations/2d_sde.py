import numpy as np
import matplotlib.pyplot as plt

#######################
# SPATIOTEMPORAL GRID #
#######################

L = 1.0  # spatial domain length
nx = 50  # number of spatial points
dx = L / (nx - 1)  # spatial step
x_grid = np.linspace(0, L, nx)

T = 1.0  # total time
nt = 500  # time steps
dt = T / nt
time_grid = np.linspace(0, T, nt + 1)

# --- Parameters: theta = [theta1, theta2, theta3] ---
theta = np.array([0.01, 2.0, 0.1])  # spatial diffusion, logistic growth, noise

# Initial condition: small perturbation in the middle
X = np.zeros((nt + 1, nx))
X[0] = 0.1 + 0.05 * np.exp(-((x_grid - 0.5) ** 2) / 0.01)


# --- Discrete Laplacian operator ---
def laplacian(u, dx):
    lap = np.zeros_like(u)
    lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return lap


# --- Euler–Maruyama simulation ---
np.random.seed(42)

for t in range(nt):
    Xt = X[t]
    lap = laplacian(Xt, dx)
    drift = theta[0] * lap + theta[1] * Xt * (1 - Xt)
    diffusion = theta[2] * Xt
    dW = np.random.normal(0.0, np.sqrt(dt), size=nx)
    X[t + 1] = Xt + drift * dt + diffusion * dW

# --- Plot final state and time evolution ---
plt.figure(figsize=(12, 5))

# Final state
plt.subplot(1, 2, 1)
plt.plot(x_grid, X[-1], label="Final state")
plt.xlabel("Space")
plt.ylabel("X(x,T)")
plt.title("Final state at time T")
plt.grid(True)

# Space-time heatmap
plt.subplot(1, 2, 2)
plt.imshow(X.T, extent=[0, T, 0, L], aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="X(x,t)")
plt.xlabel("Time")
plt.ylabel("Space")
plt.title("Spatiotemporal evolution")

plt.tight_layout()
plt.show()
