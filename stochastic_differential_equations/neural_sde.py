import numpy as np
import matplotlib.pyplot as plt


# Activation functions
def tanh(x):
    return np.tanh(x)


# Simple 1-hidden-layer neural network
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Random weights and biases (fixed for demo)
        self.W1 = np.random.randn(hidden_dim, input_dim) * 0.5
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim) * 0.5
        self.b2 = np.zeros(output_dim)

    def __call__(self, x):
        # x shape: (input_dim,) or (batch, input_dim)
        h = tanh(np.dot(self.W1, x) + self.b1)
        y = np.dot(self.W2, h) + self.b2
        return y


# Neural SDE simulation parameters
T = 1.0
dt = 0.001
N = int(T / dt)
dim = 1  # dimension of the process

# Initialize neural nets for drift and diffusion
drift_net = SimpleNN(input_dim=dim + 1, hidden_dim=10, output_dim=dim)  # input = [x, t]
diffusion_net = SimpleNN(
    input_dim=dim + 1, hidden_dim=10, output_dim=dim
)  # input = [x, t]


# Euler-Maruyama solver for the neural SDE:
def neural_sde_simulate(x0, T, dt, drift_net, diffusion_net):
    N = int(T / dt)
    xs = np.zeros(N + 1)
    xs[0] = x0
    for i in range(N):
        t = i * dt
        inp = np.array([xs[i], t])
        drift = drift_net(inp)[0]  # scalar output
        diffusion = diffusion_net(inp)[0]
        dW = np.sqrt(dt) * np.random.randn()
        xs[i + 1] = xs[i] + drift * dt + diffusion * dW
    return xs


# Simulate one path
x0 = 0.0
trajectory = neural_sde_simulate(x0, T, dt, drift_net, diffusion_net)
time = np.linspace(0, T, len(trajectory))

# Plot
plt.figure(figsize=(8, 4))
plt.plot(time, trajectory, label="Neural SDE path")
plt.xlabel("Time")
plt.ylabel("State")
plt.title("Neural SDE simulation with simple NN drift & diffusion")
plt.grid(True)
plt.legend()
plt.show()
