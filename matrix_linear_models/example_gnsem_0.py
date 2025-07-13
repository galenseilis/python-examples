import numpy as np
import matplotlib.pyplot as plt
from gnsem import GNSEM, NormalFamily, OrdinalFamily, GammaFamily, NegativeBinomialFamily, LogitLink

np.random.seed(42)

def simulate_data(n_samples=500, n_groups=3):
    # Simulate covariates
    X1 = np.random.normal(size=(n_samples, 1))       # continuous predictor
    X2 = np.random.binomial(1, 0.5, size=(n_samples, 1))  # binary predictor
    group_ids = np.random.choice(n_groups, size=n_samples)

    # Generate intermediate variables with mixed families
    eta1 = 1.0 * X1[:, 0] - 0.5 * X2[:, 0] + group_ids * 0.2
    Y1 = eta1 + np.random.normal(0, 0.5, size=n_samples)  # Normal family

    eta2 = 0.8 * Y1 - 0.2 * X2[:, 0] + 0.5
    rate2 = np.exp(eta2)
    Y2 = np.random.negative_binomial(n=2, p=2/(2+rate2))  # Negative binomial

    eta3 = 0.6 * Y2 + 0.4 * X1[:, 0]
    shape3 = 2.0
    scale3 = np.exp(eta3) / shape3
    Y3 = np.random.gamma(shape=shape3, scale=scale3)      # Gamma

    eta4 = 0.5 * Y1 + 0.5 * Y2 - 0.1 * X2[:, 0]
    thresholds = [-1.0, 0.0, 1.0]
    logits = eta4[:, None] - thresholds
    probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
    Y4 = np.array([np.random.choice(4, p=np.append(p, 1 - np.sum(p))) for p in probs[:, :3]])  # Ordinal

    X_dict = {
        'var1': np.hstack([X1, X2]),
        'var2': Y1.reshape(-1, 1),
        'var3': Y2.reshape(-1, 1),
        'var4': np.hstack([Y1.reshape(-1, 1), Y2.reshape(-1, 1), X2]),
    }

    Y_dict = {
        'var1': Y1,
        'var2': Y2,
        'var3': Y3,
        'var4': Y4,
    }

    return X_dict, Y_dict, group_ids

def main():
    X_dict, Y_dict, groups = simulate_data()

    model = GNSEM(
        structure={
            'var1': [],
            'var2': ['var1'],
            'var3': ['var2'],
            'var4': ['var1', 'var2']
        },
        families={
            'var1': NormalFamily(),
            'var2': NegativeBinomialFamily(),
            'var3': GammaFamily(),
            'var4': OrdinalFamily(n_categories=4, link=LogitLink())
        }
    )

    print("Fitting model...")
    model.fit(X_dict, Y_dict, groups)

    print("Predicting...")
    Y_pred = model.predict(X_dict, groups)

    # Plot actual vs predicted
    for var, y_true in Y_dict.items():
        if var not in Y_pred:
            continue
        y_pred = Y_pred[var]
        plt.figure(figsize=(6, 4))
        if y_true.ndim == 1:
            plt.scatter(y_true, y_pred, alpha=0.4)
        else:
            for i in range(y_true.shape[1]):
                plt.scatter(y_true[:, i], y_pred[:, i], alpha=0.4, label=f'{var}[{i}]')
        plt.xlabel(f"True {var}")
        plt.ylabel(f"Predicted {var}")
        plt.title(f"Predicted vs True for {var}")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
