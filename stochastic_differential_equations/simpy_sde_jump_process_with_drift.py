import numpy as np
import simpy
import matplotlib.pyplot as plt

# Parameters
mu = 0.1  # Drift coefficient
sigma = 0.2  # Diffusion coefficient
lambda_jump = 0.5  # Jump rate (Poisson process)
jump_mean = 0.0  # Mean of jump sizes
jump_std = 0.5  # Std of jump sizes

T = 10.0  # Total simulation time
dt = 0.01  # Time step for the SDE integration


# Main SDE model with jump handling
class JumpDiffusionSDE:
    def __init__(
        self,
        env: simpy.Environment,
        mu: float,
        sigma: float,
        lambda_jump: float,
        jump_mean: float,
        jump_std: float,
        dt: float,
    ):
        self.env = env
        self.mu = mu
        self.sigma = sigma
        self.lambda_jump = lambda_jump
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.dt = dt

        # Initial condition
        self.x = 1.0
        self.times = [0.0]
        self.values = [self.x]

        # Start both processes
        env.process(self.diffusion_process())
        env.process(self.jump_process())

    def diffusion_process(self):
        """Standard Euler-Maruyama simulation with drift + diffusion"""
        while self.env.now < T:
            dW = np.random.normal(0, np.sqrt(self.dt))
            dx = self.mu * self.x * self.dt + self.sigma * self.x * dW
            self.x += dx

            self.times.append(self.env.now + self.dt)
            self.values.append(self.x)

            yield self.env.timeout(self.dt)

    def jump_process(self):
        """SimPy-managed jump process (Poisson)"""
        while self.env.now < T:
            wait_time = np.random.exponential(1 / self.lambda_jump)
            yield self.env.timeout(wait_time)

            # Apply jump
            jump = np.random.normal(self.jump_mean, self.jump_std)
            self.x += jump

            # Log jump time and new value
            self.times.append(self.env.now)
            self.values.append(self.x)


# Run the simulation
env = simpy.Environment()
sde = JumpDiffusionSDE(env, mu, sigma, lambda_jump, jump_mean, jump_std, dt)
env.run(until=T)

# Sort values chronologically in case jump and diffusion overlap
sorted_indices = np.argsort(sde.times)
t_sorted = np.array(sde.times)[sorted_indices]
x_sorted = np.array(sde.values)[sorted_indices]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(t_sorted, x_sorted, label="Jump-Diffusion SDE", linewidth=2)
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("Jump-Diffusion SDE Simulated with SimPy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
