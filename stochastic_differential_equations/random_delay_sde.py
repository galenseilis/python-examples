import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Parameters
T = 10.0  # Total simulation time
dt = 0.01  # Time step
tau_max = 1.0  # Maximum delay
alpha = 0.5  # Drift scaling
beta = 0.2  # Diffusion scaling

# Time grid
num_steps = int(T / dt)
delay_buffer_steps = int(tau_max / dt)
total_steps = num_steps + delay_buffer_steps
time = np.linspace(-tau_max, T, total_steps + 1)

# Initial history function φ(t) = 1.0 for t ∈ [-τ_max, 0]
x = np.ones(total_steps + 1)

# Pre-compute Brownian increments
dW = np.random.normal(0.0, np.sqrt(dt), size=num_steps)


# Helper: get interpolated delayed value
def get_delayed_value(t_current, x_vals, time_vals):
    tau = np.random.uniform(0.0, tau_max)
    t_delay = t_current - tau
    if t_delay < -tau_max:
        raise ValueError("Delay exceeds defined history range")
    interpolator = interp1d(time_vals, x_vals, fill_value="extrapolate")
    return interpolator(t_delay)


# Simulate SDDE with random delay
for i in range(delay_buffer_steps, total_steps):
    t = time[i]
    x_delayed = get_delayed_value(t, x[: i + 1], time[: i + 1])
    drift = alpha * x_delayed
    diffusion = beta * x_delayed
    x[i + 1] = x[i] + drift * dt + diffusion * dW[i - delay_buffer_steps]

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(time, x, label="SDDE with random delay")
plt.axvline(0, color="gray", linestyle="--", label="t = 0")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title("Stochastic Delay Differential Equation with Random Delay")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
