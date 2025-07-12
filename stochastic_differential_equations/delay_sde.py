import numpy as np
import matplotlib.pyplot as plt

# Parameters
T = 10.0  # Total time
dt = 0.01  # Time step
tau = 1.0  # Delay
alpha = 0.5  # Drift coefficient
beta = 0.2  # Diffusion coefficient

# Time grid
num_steps = int(T / dt)
delay_steps = int(tau / dt)
time = np.linspace(-tau, T, num_steps + delay_steps + 1)

# Initial history function φ(t) = 1.0 for t in [-τ, 0]
x = np.ones(num_steps + delay_steps + 1)

# Brownian increments
dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=num_steps)

# Euler–Maruyama simulation with delay
for i in range(delay_steps, delay_steps + num_steps):
    x_delayed = x[i - delay_steps]  # x(t - τ)
    drift = alpha * x_delayed
    diffusion = beta * x_delayed
    x[i + 1] = x[i] + drift * dt + diffusion * dW[i - delay_steps]

# Plot the result
plt.figure(figsize=(10, 5))
plt.plot(time, x, label="SDDE solution")
plt.axvline(x=0, color="gray", linestyle="--", label="Delay boundary")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("Stochastic Delay Differential Equation (SDDE)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
