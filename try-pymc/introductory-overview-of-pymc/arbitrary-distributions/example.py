import pymc as pm
import pytensor.tensor as at

with pm.Model() as model:
	alpha = pm.Uniform("intercept", -100, 100)

	# Create custom densities
	beta = pm.DensityDist(
		"beta", 
		logp=lambda value: -1.5 * at.log(1 + value**2)
		)
	eps = pm.DensityDist(
		"eps",
		logp=lambda value: -at.log(at.abs_(value))
		)

	# Create likelihood
	like = pm.Normal(
		'y_est',
		mu=alpha + beta * X,
		sigma=eps,
		observed=Y
		)
