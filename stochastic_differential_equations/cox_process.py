import numpy as np
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ---------------- #
n = 3  # state dimension
T = 10.0
dt = 0.01
nt = int(T / dt)
time = np.linspace(0, T, nt + 1)

# System matrix (linear ODE)
A = np.array([[-0.1, 0.0, 0.05], [0.0, -0.2, 0.1], [0.0, 0.0, -0.1]])

# Initial state
x0 = np.array([1.0, 0.5, -0.2])

# Tunable parameter vector (controls intensity)
theta = np.array([1.0, -1.0, 0.5])

# ---------------- SIMULATE LATENT STATE ---------------- #
X = np.zeros((nt + 1, n))
X[0] = x0

for t in range(nt):
    X[t + 1] = X[t] + dt * (A @ X[t])

# ---------------- COMPUTE INTENSITY ---------------- #
lambda_t = np.exp(X @ theta)  # λ(t) = exp(θᵀ x(t))

# ---------------- SIMULATE EVENTS ---------------- #
events = []
for t in range(nt):
    rate = lambda_t[t]
    prob = rate * dt
    if np.random.rand() < prob:
        events.append(time[t])

events = np.array(events)

# ---------------- PLOTTING ---------------- #
plt.figure(figsize=(12, 6))

# 1. Intensity over time
plt.subplot(2, 1, 1)
plt.plot(time, lambda_t, label="Intensity λ(t)", color="darkred")
plt.xlabel("Time")
plt.ylabel("λ(t)")
plt.title("Poisson Intensity Over Time")
plt.grid(True)
plt.legend()

# 2. Spike train of Poisson events
plt.subplot(2, 1, 2)
plt.eventplot(events, orientation="horizontal", colors="black", lineoffsets=0.5)
plt.xlim([0, T])
plt.ylim([0, 1])
plt.xlabel("Time")
plt.yticks([])
plt.title("Poisson Events (Generated from Intensity)")
plt.grid(True)

plt.tight_layout()
plt.show()
