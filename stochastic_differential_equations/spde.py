import numpy as np
import matplotlib.pyplot as plt

# ----------- PARAMETERS ----------- #
theta = 0.01  # diffusion coefficient
sigma = 0.1  # noise strength

L = 1.0  # spatial domain length
nx = 100  # number of spatial points
dx = L / (nx - 1)
x = np.linspace(0, L, nx)

T = 1.0  # total time
nt = 500  # number of time steps
dt = T / nt
time = np.linspace(0, T, nt + 1)

# ----------- INITIAL CONDITION ----------- #
X = np.zeros((nt + 1, nx))
X[0] = np.sin(np.pi * x)  # initial profile


# ----------- BOUNDARY CONDITIONS ----------- #
def apply_boundary_conditions(u):
    u[0] = 0.0  # Dirichlet at left
    u[-1] = 0.0  # Dirichlet at right
    return u


# ----------- LAPLACIAN OPERATOR ----------- #
def laplacian(u, dx):
    lap = np.zeros_like(u)
    lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
    return lap


# ----------- SIMULATION ----------- #
np.random.seed(42)

for t in range(nt):
    Xt = X[t]
    lap = laplacian(Xt, dx)

    # Noise: approximated space-time white noise
    noise = np.random.normal(0, np.sqrt(dt / dx), size=nx)
    dXt = theta * lap * dt + sigma * noise
    X[t + 1] = Xt + dXt
    X[t + 1] = apply_boundary_conditions(X[t + 1])

# ----------- PLOTTING ----------- #
plt.figure(figsize=(12, 5))

# Final state
plt.subplot(1, 2, 1)
plt.plot(x, X[-1], label="Final state")
plt.xlabel("Space")
plt.ylabel("X(x, T)")
plt.title("Final state at time T")
plt.grid(True)

# Heatmap of spatiotemporal evolution
plt.subplot(1, 2, 2)
plt.imshow(X.T, extent=[0, T, 0, L], aspect="auto", origin="lower", cmap="plasma")
plt.colorbar(label="X(x, t)")
plt.xlabel("Time")
plt.ylabel("Space")
plt.title("Spatiotemporal Evolution (Stochastic Heat Equation)")

plt.tight_layout()
plt.show()
