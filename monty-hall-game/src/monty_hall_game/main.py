import pymc as pm
import numpy as np
import arviz as az
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import cloudpickle

# Define the levels for the decision dimension and simulated data
decision_levels = ["stick", "switch"]

df = pd.read_csv('example.csv')

# Simulated data: binary outcomes (0 or 1), and corresponding decisions (0 = stick, 1 = switch)
outcome = df.final_win  # 1 = success, 0 = failure
decisions = df.decision  # Random decisions (0 = stick, 1 = switch)

with pm.Model() as model:
    # Add a coordinate for the decision dimension
    model.add_coord("decision", decision_levels)
    
    # Define Beta priors for the success probabilities for "stick" and "switch"
    p = pm.Beta("p", alpha=2, beta=2, dims="decision")
    
    # Define the observed decisions (0 = stick, 1 = switch)
    decision_obs = pm.Data("decision_obs", decisions)
    
    # Likelihood: use the observed decision to index into p
    obs = pm.Bernoulli("obs", p=p[decision_obs], observed=outcome)
    
    # Sample from the posterior
    trace = pm.sample(1000, return_inferencedata=True)

trace.to_netcdf("trace.nc")
pickle_filepath = f'model.pkl'
dict_to_save = {'model': model}

with open(pickle_filepath , 'wb') as buff:
    cloudpickle.dump(dict_to_save, buff)
