import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# --- Parameters ---
T = 10.0
dt = 0.01
tau_max = 1.0
alpha_true = 0.5
beta_true = 0.3
obs_noise_std = 0.1

num_steps = int(T / dt)
delay_steps = int(tau_max / dt)
total_steps = num_steps + delay_steps
time = np.linspace(-tau_max, T, total_steps + 1)

# --- Initial condition ---
x_true = np.ones(total_steps + 1)

# --- Generate fixed Brownian increments ---
dW = np.random.normal(0, np.sqrt(dt), size=num_steps)

# --- Pre-generate fixed random delays for drift and diffusion ---
np.random.seed(42)  # for reproducibility of delays and noise only
tau_drift_seq = np.random.uniform(0, tau_max, size=num_steps)
tau_diff_seq = np.random.uniform(0, tau_max, size=num_steps)


# --- Interpolator helper ---
def get_delayed_value(t, x_vals, time_vals):
    interp = interp1d(time_vals, x_vals, fill_value="extrapolate")
    return interp(t)


# --- Simulate the true system with fixed delays ---
for i in range(delay_steps, total_steps):
    t = time[i]
    tau_drift = tau_drift_seq[i - delay_steps]
    tau_diff = tau_diff_seq[i - delay_steps]

    t_drift = t - tau_drift
    t_diff = t - tau_diff

    x_delay_drift = get_delayed_value(t_drift, x_true[: i + 1], time[: i + 1])
    x_delay_diff = get_delayed_value(t_diff, x_true[: i + 1], time[: i + 1])

    drift = alpha_true * x_delay_drift
    diffusion = beta_true * x_delay_diff

    x_true[i + 1] = x_true[i] + drift * dt + diffusion * dW[i - delay_steps]

# --- Noisy observations ---
observed_data = x_true[delay_steps : delay_steps + num_steps] + np.random.normal(
    0, obs_noise_std, size=num_steps
)


# --- Model simulation using the fixed delays ---
def simulate_model(params, dW, tau_drift_seq, tau_diff_seq, return_full=False):
    alpha, beta = params
    x_sim = np.ones(total_steps + 1)

    for i in range(delay_steps, total_steps):
        t = time[i]
        tau_drift = tau_drift_seq[i - delay_steps]
        tau_diff = tau_diff_seq[i - delay_steps]

        t_drift = t - tau_drift
        t_diff = t - tau_diff

        x_delay_drift = get_delayed_value(t_drift, x_sim[: i + 1], time[: i + 1])
        x_delay_diff = get_delayed_value(t_diff, x_sim[: i + 1], time[: i + 1])

        drift = alpha * x_delay_drift
        diffusion = beta * x_delay_diff

        x_sim[i + 1] = x_sim[i] + drift * dt + diffusion * dW[i - delay_steps]

    return x_sim if return_full else x_sim[delay_steps : delay_steps + num_steps]


# --- Loss function ---
def loss_fn(params):
    x_sim = simulate_model(params, dW, tau_drift_seq, tau_diff_seq)
    return np.mean((x_sim - observed_data) ** 2)


# --- Fit ---
initial_guess = [0.1, 0.1]
bounds = [(0, 2), (0, 2)]
result = minimize(loss_fn, initial_guess, bounds=bounds, method="L-BFGS-B")

alpha_fit, beta_fit = result.x
print(f"Fitted parameters: alpha = {alpha_fit:.4f}, beta = {beta_fit:.4f}")

# --- Simulate fitted trajectory ---
x_fit = simulate_model(
    [alpha_fit, beta_fit], dW, tau_drift_seq, tau_diff_seq, return_full=True
)

# --- Plot results ---
plt.figure(figsize=(12, 5))
plt.plot(
    time[delay_steps : delay_steps + num_steps],
    observed_data,
    label="Observed (noisy)",
    linestyle="--",
    alpha=0.6,
)
plt.plot(time, x_true, label="True trajectory", linewidth=2)
plt.plot(time, x_fit, label="Fitted trajectory", linewidth=2, linestyle="-.")
plt.axvline(0, color="gray", linestyle="--", label="t=0")
plt.xlabel("Time")
plt.ylabel("x(t)")
plt.title(
    "SDDE with Fixed Random Delays in Drift and Diffusion\nFitting α and β parameters"
)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
