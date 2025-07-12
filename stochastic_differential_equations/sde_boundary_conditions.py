import numpy as np
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ---------------- #
theta = 0.01  # diffusion coefficient
sigma = 0.1  # noise intensity
L = 1.0  # spatial domain length
nx = 50  # number of spatial points
dx = L / (nx - 1)
x_grid = np.linspace(0, L, nx)

T = 1.0  # total simulation time
nt = 500  # number of time steps
dt = T / nt
time_grid = np.linspace(0, T, nt + 1)

# ---------------- INITIAL CONDITION ---------------- #
X = np.zeros((nt + 1, nx))
X[0] = np.sin(np.pi * x_grid)  # initial profile

# ---------------- BOUNDARY CONDITION ---------------- #
boundary_type = "dirichlet"  # choose: 'dirichlet', 'neumann', or 'periodic'

# Set custom values for Dirichlet BCs (only used if type is 'dirichlet')
left_bc_value = 0.0
right_bc_value = 0.0


# ---------------- LAPLACIAN ---------------- #
def laplacian(u, dx, bc_type):
    lap = np.zeros_like(u)
    if bc_type == "dirichlet":
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2
        # Boundaries will be enforced outside the laplacian

    elif bc_type == "neumann":
        # zero Neumann (∂X/∂x = 0) at both ends
        lap[0] = (u[1] - u[0]) / dx**2
        lap[-1] = (u[-2] - u[-1]) / dx**2
        lap[1:-1] = (u[2:] - 2 * u[1:-1] + u[:-2]) / dx**2

    elif bc_type == "periodic":
        lap = (np.roll(u, -1) - 2 * u + np.roll(u, 1)) / dx**2

    else:
        raise ValueError("Unknown boundary condition type.")
    return lap


# ---------------- SIMULATION ---------------- #
np.random.seed(42)

for t in range(nt):
    Xt = X[t]
    lap = laplacian(Xt, dx, boundary_type)
    dW = np.random.normal(0.0, np.sqrt(dt), size=nx)
    X[t + 1] = Xt + theta * lap * dt + sigma * dW

    # Enforce boundary conditions explicitly
    if boundary_type == "dirichlet":
        X[t + 1, 0] = left_bc_value
        X[t + 1, -1] = right_bc_value
    elif boundary_type == "periodic":
        X[t + 1, 0] = X[t + 1, -2]
        X[t + 1, -1] = X[t + 1, 1]
    # For Neumann, already handled in laplacian

# ---------------- PLOTTING ---------------- #
plt.figure(figsize=(12, 5))

# Final spatial profile
plt.subplot(1, 2, 1)
plt.plot(x_grid, X[-1], label="X(x, T)")
plt.xlabel("Space")
plt.ylabel("X(x, T)")
plt.title(f"Final state with {boundary_type} BC")
plt.grid(True)

# Heatmap over space-time
plt.subplot(1, 2, 2)
plt.imshow(X.T, extent=[0, T, 0, L], aspect="auto", origin="lower", cmap="viridis")
plt.colorbar(label="X(x, t)")
plt.xlabel("Time")
plt.ylabel("Space")
plt.title("Spatiotemporal Evolution")

plt.tight_layout()
plt.show()
