import numpy as np
import queueing_tool as qt

# Define the parameters
arrival_rate = 10.0
service_rate = 4.0
num_servers = 3
sim_time = 800
warmup_time = 100

LAMBDA = 1.0 / arrival_rate
MU = 1.0 / service_rate

# Make an adjacency list
adjacency_list = {0: {1: {}}, 1: {}}

# Make an object that has the same dimensions as your adjacency list that
# specifies the type of queue along each edge.
edge_list = {0: {1: 1}}

# Create a networkx directed graph using the adjacency list and edge list
g = qt.adjacency2graph(adjacency=adjacency_list, edge_type=edge_list)

# Make a mapping between the edge types and the queue classes that sit on each
# edge. Do not use 0 as a key, it's used to map to NullQueues.
q_classes = {0: qt.NullQueue, 1: qt.QueueServer}


# Define the arrival and service functions
def arrival_func(t):
    return t + np.random.exponential(LAMBDA)


def service_func(t):
    return t + np.random.exponential(MU)


# Make a mapping between the edge types and the parameters used to make those
# queues.
q_args = {
    1: {
        "arrival_f": arrival_func,
        "service_f": service_func,
        "num_servers": num_servers,
    },
}

# Put it all together to create the network
qn = qt.QueueNetwork(g=g, q_classes=q_classes, q_args=q_args)

# Set the maximum number of agents
qn.max_agents = np.inf

# Initialize the network
qn.initialize(edge_type=1)

# Data is not collected by default. This makes all queues collect data as the
# simulations take place.
qn.start_collecting_data()

# Simulate the network
qn.simulate(t=warmup_time + sim_time)

# Collect data
data = qn.get_queue_data()

# Calculate the average waiting time
data = data[:, :2]
fin_data = data[data[:, 1] > 0]
fin_wait = fin_data[:, 1] - fin_data[:, 0]
fin_wait.mean()
average_waiting_time = fin_wait.mean()
