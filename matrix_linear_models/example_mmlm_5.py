import numpy as np
from mmlm import MatrixLinearMixedEffectsModel

# Recreate synthetic dataset
np.random.seed(1234)
n_samples_per_group = 50
n_features = 2
n_outputs = 2
n_groups = 3

# Fixed effects
A_fixed = np.array([[1.5, -0.5], [-1.0, 2.0]])
B_fixed = np.array([0.2, -0.5])

# Random effects
A_random = np.array(
    [[[0.0, 0.0], [0.0, 0.0]], [[1.0, -0.5], [-0.5, 1.0]], [[-1.5, 0.7], [1.0, -1.5]]]
)
B_random = np.array([[0.0, 0.0], [1.0, -1.0], [-2.0, 2.0]])

# Generate data
groups = np.repeat(np.arange(n_groups), n_samples_per_group)
X = np.empty((n_samples_per_group * n_groups, n_features))
for g in range(n_groups):
    mask = groups == g
    X[mask, 0] = np.linspace(0, 5, n_samples_per_group)
    X[mask, 1] = np.linspace(5, 0, n_samples_per_group)

noise_std = 0.3
Y = np.empty((n_samples_per_group * n_groups, n_outputs))
for i in range(len(groups)):
    g = groups[i]
    x_i = X[i]
    Y[i] = (
        (A_fixed + A_random[g]) @ x_i
        + B_fixed
        + B_random[g]
        + noise_std * np.random.randn(n_outputs)
    )

# Fit model
model = MatrixLinearMixedEffectsModel()
model.fit(X, Y, groups)

# Check intercepts
print("\n=== Intercept Check ===")
for j in range(n_outputs):
    fixed = model.B_[j]
    group_intercepts = model.B_[j] + model.Bg_[:, j]
    min_group = np.min(group_intercepts)
    max_group = np.max(group_intercepts)

    print(f"Output {j}:")
    print(f"  Fixed intercept B[{j}] = {fixed:.3f}")
    print(f"  Group-specific intercepts: {group_intercepts}")
    print(f"  Range: [{min_group:.3f}, {max_group:.3f}]")
    print(
        "  ✅ OK" if min_group <= fixed <= max_group else "  ❌ Outside expected range"
    )
    print()

# Check slopes
print("\n=== Slope Check ===")
for j in range(n_outputs):
    for k in range(n_features):
        fixed = model.A_[j, k]
        group_slopes = model.A_[j, k] + model.Ag_[:, j, k]
        min_group = np.min(group_slopes)
        max_group = np.max(group_slopes)

        print(f"Slope A[{j},{k}]:")
        print(f"  Fixed slope = {fixed:.3f}")
        print(f"  Group-specific slopes: {group_slopes}")
        print(f"  Range: [{min_group:.3f}, {max_group:.3f}]")
        print(
            "  ✅ OK"
            if min_group <= fixed <= max_group
            else "  ❌ Outside expected range"
        )
        print()
