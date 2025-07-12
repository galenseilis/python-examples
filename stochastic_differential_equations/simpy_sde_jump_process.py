import numpy as np
import simpy
import matplotlib.pyplot as plt

# Parameters
mu = 0.1  # Drift
sigma = 0.2  # Diffusion coefficient
lambda_jump = 0.5  # Jump rate (Poisson)
jump_mean = 0.0
jump_std = 1.0
T = 10.0  # Total simulation time
dt = 0.01  # Time step for SDE


# State holder
class JumpSDE:
    def __init__(self, env, mu, sigma, jump_rate, jump_mean, jump_std, dt):
        self.env = env
        self.mu = mu
        self.sigma = sigma
        self.jump_rate = jump_rate
        self.jump_mean = jump_mean
        self.jump_std = jump_std
        self.dt = dt

        self.t_values = [0.0]
        self.x_values = [0.0]  # Initial condition: X(0) = 0
        self.current_x = 0.0

        # Start both processes
        env.process(self.diffusion_process())
        env.process(self.jump_process())

    def diffusion_process(self):
        while self.env.now < T:
            # Euler-Maruyama integration for the diffusion part
            dW = np.random.normal(0, np.sqrt(self.dt))
            dx = self.mu * self.current_x * self.dt + self.sigma * dW
            self.current_x += dx

            self.t_values.append(self.env.now + self.dt)
            self.x_values.append(self.current_x)

            yield self.env.timeout(self.dt)

    def jump_process(self):
        while self.env.now < T:
            # Wait for the next Poisson jump time
            wait_time = np.random.exponential(1.0 / self.jump_rate)
            yield self.env.timeout(wait_time)

            # Apply jump
            jump_size = np.random.normal(self.jump_mean, self.jump_std)
            self.current_x += jump_size

            # Log state at jump time
            self.t_values.append(self.env.now)
            self.x_values.append(self.current_x)


# Set up SimPy environment and run
env = simpy.Environment()
sde = JumpSDE(env, mu, sigma, lambda_jump, jump_mean, jump_std, dt)
env.run(until=T)

# Sort times and values in case of out-of-order events
sorted_indices = np.argsort(sde.t_values)
t_sorted = np.array(sde.t_values)[sorted_indices]
x_sorted = np.array(sde.x_values)[sorted_indices]

# Plot the process
plt.figure(figsize=(10, 5))
plt.plot(t_sorted, x_sorted, label="Jump-Diffusion Process")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.title("SDE with Random Jumps (SimPy-based Simulation)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
