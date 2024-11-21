import pymc as pm
import pytensor.tensor as pt
import networkx as nx
import numpy as np
import arviz as az


# Function to create the PyMC model from a NetworkX tree
def create_hierarchical_model(tree, data):
    # Ensure the tree is a directed acyclic graph
    assert nx.is_tree(tree), "The input must be a tree (a directed acyclic graph)."

    # Root node (this is the highest level in the hierarchy)
    root = [n for n, d in tree.in_degree() if d == 0][0]

    with pm.Model() as model:
        # Define the root level parameters with non-centered parameterization
        mu_root_raw = pm.Normal(f"mu_{root}_raw", mu=0, sigma=1)
        sigma_root = pm.Exponential(f"sigma_{root}", 1.0)
        mu_root = pm.Deterministic(f"mu_{root}", mu_root_raw * sigma_root)

        # Recursive function to traverse the tree and create the hierarchy
        def add_nodes(node):
            mu_parent = model[f"mu_{node}"]
            sigma_parent = model[f"sigma_{node}"]

            for child in tree.successors(node):
                # Define the hierarchical relationship for each child with non-centered parameterization
                mu_child_raw = pm.Normal(f"mu_{child}_raw", mu=0, sigma=1)
                sigma_child = pm.Exponential(f"sigma_{child}", 1.0)
                mu_child = pm.Deterministic(
                    f"mu_{child}", mu_parent + mu_child_raw * sigma_child
                )

                # Add the child node recursively
                add_nodes(child)

        # Start from the root node
        add_nodes(root)

        # Assume data is observed at the leaf nodes of the tree
        for leaf in [n for n in tree.nodes() if tree.out_degree(n) == 0]:
            # Observations at the leaf node level
            obs = pm.Normal(
                f"obs_{leaf}",
                mu=model[f"mu_{leaf}"],
                sigma=model[f"sigma_{leaf}"],
                observed=data[leaf],
            )

    return model


# Creating a sample tree with networkx
G = nx.DiGraph()
G.add_edges_from(
    [
        ("root", "group1"),
        ("root", "group2"),
        ("group1", "subgroup1"),
        ("group1", "subgroup2"),
        ("group1", "subgroup3"),
        ("group2", "subgroup4"),
        ("group2", "subgroup5"),
    ]
)

# Simulated data generation based on the hierarchy
np.random.seed(42)  # For reproducibility

# Root level
mu_root = 0
sigma_root = 2

# Group levels
mu_group1 = mu_root + np.random.normal(0, sigma_root)
mu_group2 = mu_root + np.random.normal(0, sigma_root)
sigma_group = 1.5

# Subgroup levels
mu_subgroup1 = mu_group1 + np.random.normal(0, sigma_group)
mu_subgroup2 = mu_group1 + np.random.normal(0, sigma_group)
mu_subgroup3 = mu_group1 + np.random.normal(0, sigma_group)
mu_subgroup4 = mu_group2 + np.random.normal(0, sigma_group)
mu_subgroup5 = mu_group2 + np.random.normal(0, sigma_group)
sigma_subgroup = 1.0

# Observations at the leaf nodes
data = {
    "subgroup1": np.random.normal(mu_subgroup1, sigma_subgroup, 100),
    "subgroup2": np.random.normal(mu_subgroup2, sigma_subgroup, 100),
    "subgroup3": np.random.normal(mu_subgroup3, sigma_subgroup, 100),
    "subgroup4": np.random.normal(mu_subgroup4, sigma_subgroup, 100),
    "subgroup5": np.random.normal(mu_subgroup5, sigma_subgroup, 100),
}

# Create the PyMC model
hierarchical_model = create_hierarchical_model(G, data)

# Perform inference with increased target_accept
with hierarchical_model:
    trace = pm.sample(1000, tune=1000, target_accept=0.99, return_inferencedata=True)

# Adjust ArviZ settings to allow more subplots
az.rcParams["plot.max_subplots"] = 30

# Trace plot to check for convergence
az.plot_trace(trace)

# Energy plot to check for sampling issues
az.plot_energy(trace)

# Summary of the trace to check for convergence diagnostics
summary = az.summary(trace)
print(summary)
