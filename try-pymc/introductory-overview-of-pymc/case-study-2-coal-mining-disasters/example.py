import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm

# Define data
disaster_data = pd.Series(
    [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
    3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
    2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
    1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
    0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
    3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
)

years = np.arange(1851, 1962)

# Plot data

plt.plot(years, disaster_data, "o", markersize=8, alpha=0.4)
plt.ylabel("Disaster count")
plt.xlabel("Year")
plt.savefig("disaster_data_plot.png", dpi=300)
plt.close()

# Model specification
with pm.Model() as disaster_model:
	switchpoint = pm.DiscreteUniform(
		"switchpoint", 
		lower=years.min(),
		upper=years.max(),
		)
	
	# Priors for pre/post switch rates of number of disasters.
	early_rate = pm.Exponential('early_rate', 1.0)
	later_rate = pm.Exponential('later_rate', 1.0)

	# Allocate Poisson rates to years before/after
	rate = pm.math.switch(
		switchpoint >= years, 
		early_rate, 
		later_rate
		)

	disasters = pm.Poisson(
		'disasters', 
		rate, 
		observed=disaster_data
		)


# Sample from the model
with disaster_model:
	idata = pm.sample(10_000)


# Posterior trace plot
axes_arr = az.plot_trace(idata)
plt.draw()
for ax in axes_arr.flatten():
	if ax.get_title() == 'switchpoint':
		labels=[label.get_text() for label in ax.get_xticklabels()]
		ax.set_xticklabels(labels, rotation=45, ha='right')
		break
plt.draw()
plt.tight_layout()
plt.savefig('posterior_trace.png', dpi=300)
plt.close()

# Plot number of accidents
plt.figure(figsize=(10,8))
plt.plot(years, disaster_data, '.', alpha=0.6)
plt.ylabel("Number of accidents", fontsize=16)
plt.xlabel("Year", fontsize=16)

trace = idata.posterior.stack(draws=("chain", "draw"))

plt.vlines(
	trace['switchpoint'].mean(),
	disaster_data.min(),
	disaster_data.max(),
	color="C1"
	)

average_disasters = np.zeros_like(disaster_data, dtype='float')

for i, year in enumerate(years):
	idx = year < trace['switchpoint']
	average_disasters[i] = np.mean(
		np.where(idx, trace['early_rate'], trace['later_rate'])
		)

sp_hpd = az.hdi(idata, var_names=['switchpoint'])['switchpoint'].values

plt.fill_betweenx(
	y=[disaster_data.min(), disaster_data.max()],
	x1=sp_hpd[0],
	x2=sp_hpd[1],
	alpha=0.5,
	color="C1"
	)

plt.plot(years, average_disasters, 'k--', lw=2)
plt.tight_layout()
plt.savefig('switch_point_plot.png', dpi=300)
plt.close()
