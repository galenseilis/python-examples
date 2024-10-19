import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('Solarize_Light2')

np.random.seed(2018)

SAMPLE_SIZE = 100
LAMBDA = 10

x = pd.date_range('2024-02-01', '2024-02-21')
demand = np.random.poisson(lam=LAMBDA, size=x.size)

weight_counts = {
	'Met':demand * ((demand - LAMBDA) <= 0) + LAMBDA * ((demand - LAMBDA) > 0),
	'Unmet':(demand - LAMBDA) * ((demand - LAMBDA) > 0)
}

fig, ax = plt.subplots()
bottom = np.zeros(x.size)

for boolean, weight_count in weight_counts.items():
	p = ax.bar(x, weight_count, width=1, label=boolean, bottom=bottom, alpha=0.5)
	bottom += weight_count

ax.axhline(y=10, color='r', linestyle='-', label='Available ORs')

ax.set_xlabel('Date')
ax.set_ylabel('Required ORs')

ax.set_title('Avg. Demand = Avg. Capacity',c='gray')
ax.legend(loc='lower left')
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('needed_ors_on_avg.png', dpi=300, transparent=True)
plt.close()
