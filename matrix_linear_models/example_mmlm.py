import numpy as np
from mmlm import MatrixLinearMixedEffectsModel

np.random.seed(0)

# Generate synthetic data with groups
n_samples = 300
n_features = 3
n_outputs = 2
n_groups = 4

# Fixed effects
A_fixed = np.array([[1.0, 0.5, -0.5], [-0.5, 1.5, 0.3]])
B_fixed = np.array([0.1, -0.2])

# Random effects per group
A_random = 0.2 * np.random.randn(n_groups, n_outputs, n_features)
B_random = 0.1 * np.random.randn(n_groups, n_outputs)

# Assign groups randomly
groups = np.random.choice(n_groups, size=n_samples)

# Generate features
X = np.random.randn(n_samples, n_features)

# Generate outputs with mixed effects + noise
Y = np.empty((n_samples, n_outputs))
for i in range(n_samples):
    g = groups[i]
    Y[i] = (
        (A_fixed + A_random[g]) @ X[i]
        + B_fixed
        + B_random[g]
        + 0.05 * np.random.randn(n_outputs)
    )

# Fit model
model = MatrixLinearMixedEffectsModel()
model.fit(X, Y, groups)

print("Fixed slope estimates (A):\n", model.A_)
print("Fixed intercept estimates (B):\n", model.B_)

print("Random slope effects (per group):\n", model.Ag_)
print("Random intercept effects (per group):\n", model.Bg_)

# Predict and score
Y_pred = model.predict(X, groups)
print("R^2 score:", model.score(X, Y, groups))
