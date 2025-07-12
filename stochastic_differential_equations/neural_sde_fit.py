import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# --- Activation and NN class from before, but exposing parameters ---
def tanh(x):
    return np.tanh(x)


class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.1
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.1
        self.b2 = np.zeros(output_dim)

    def forward(self, x):
        h = tanh(self.W1 @ x + self.b1)
        y = self.W2 @ h + self.b2
        return y

    def get_params(self):
        return [self.W1, self.b1, self.W2, self.b2]

    def set_params(self, params):
        self.W1, self.b1, self.W2, self.b2 = params


# Euler-Maruyama simulation of neural SDE
def neural_sde_simulate(x0, T, dt, drift_net, diffusion_net, noise_seed=None):
    if noise_seed is not None:
        np.random.seed(noise_seed)
    N = int(T / dt)
    xs = np.zeros(N + 1)
    xs[0] = x0
    for i in range(N):
        t = i * dt
        inp = np.array([xs[i], t])
        drift = drift_net.forward(inp)[0]
        diffusion = diffusion_net.forward(inp)[0]
        dW = np.sqrt(dt) * np.random.randn()
        xs[i + 1] = xs[i] + drift * dt + diffusion * dW
    return xs


# Mean squared error loss
def loss_fn(simulated, observed):
    return np.mean((simulated - observed) ** 2)


# Numerical gradient estimation of loss w.r.t single param tensor
def numerical_grad(param_tensor, index, loss_func, epsilon=1e-5):
    original_value = param_tensor.flat[index]
    param_tensor.flat[index] = original_value + epsilon
    plus_loss = loss_func()
    param_tensor.flat[index] = original_value - epsilon
    minus_loss = loss_func()
    param_tensor.flat[index] = original_value
    return (plus_loss - minus_loss) / (2 * epsilon)


# Training step: update all parameters by gradient descent
def train_step(drift_net, diffusion_net, x0, T, dt, observed, lr=1e-2, noise_seed=0):
    # Closure to compute loss with current parameters
    def compute_loss():
        sim = neural_sde_simulate(
            x0, T, dt, drift_net, diffusion_net, noise_seed=noise_seed
        )
        return loss_fn(sim, observed)

    current_loss = compute_loss()

    # Update params for drift_net and diffusion_net
    for net in [drift_net, diffusion_net]:
        params = net.get_params()
        grads = []
        for p in params:
            grad = np.zeros_like(p)
            # compute numerical gradient elementwise
            for idx in range(p.size):
                grad.flat[idx] = numerical_grad(p, idx, compute_loss)
            grads.append(grad)

        # Gradient descent step
        for p, g in zip(params, grads):
            p -= lr * g

        net.set_params(params)


# --- Generate synthetic data with known neural nets ---

np.random.seed(42)
T = 1.0
dt = 0.01
x0 = 0.0

# "True" neural nets (fixed weights)
true_drift_net = SimpleNN(2, 10, 1)
true_diffusion_net = SimpleNN(2, 10, 1)

# Manually fix parameters to known values for stable data generation
for p in true_drift_net.get_params():
    p[:] = np.random.randn(*p.shape) * 0.2
for p in true_diffusion_net.get_params():
    p[:] = np.abs(np.random.randn(*p.shape)) * 0.1  # positive diffusion

# Generate one noisy trajectory as training data
observed_traj = neural_sde_simulate(
    x0, T, dt, true_drift_net, true_diffusion_net, noise_seed=123
)

# --- Initialize model nets to random params ---
model_drift_net = SimpleNN(2, 10, 1)
model_diffusion_net = SimpleNN(2, 10, 1)

# --- Training ---
epochs = 1000
for epoch in tqdm(range(epochs)):
    train_step(
        model_drift_net,
        model_diffusion_net,
        x0,
        T,
        dt,
        observed_traj,
        lr=1e-2,
        noise_seed=123,
    )

# --- Compare simulated path after training ---
simulated_after = neural_sde_simulate(
    x0, T, dt, model_drift_net, model_diffusion_net, noise_seed=123
)

# Plot
time = np.linspace(0, T, int(T / dt) + 1)
plt.figure(figsize=(10, 5))
plt.plot(time, observed_traj, label="Observed trajectory (true)")
plt.plot(
    time, simulated_after, label="Model trajectory after training", linestyle="dashed"
)
plt.xlabel("Time")
plt.ylabel("State")
plt.title("Fitting Neural SDE to Data (Finite Differences)")
plt.legend()
plt.grid(True)
plt.show()
