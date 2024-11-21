import pandas as pd
from prophet import Prophet
import numpy as np

# Create a sample dataframe
df = pd.DataFrame(
    {
        "ds": pd.date_range(start="2024-01-01", end="2024-01-31"),
        "y": range(31),  # Sample target variable
        "regressor_1": [i % 5 for i in range(31)],  # Sample regressor 1
        "regressor_2": [i % 7 for i in range(31)],  # Sample regressor 2
    }
)

# Introduce missing values in regressor_2
import random

missing_indices = random.sample(range(len(df)), k=5)
df.loc[missing_indices, "regressor_2"] = np.nan

# Initialize Prophet model with additional regressors
model = Prophet()
model.add_regressor("regressor_1")
model.add_regressor("regressor_2")

# Handle missing values in additional regressors
df["regressor_2"] = df["regressor_2"].interpolate(method="linear")

# Fit the model
model.fit(df)

# Create future dataframe for prediction
future = model.make_future_dataframe(periods=7)

# Fill in missing values for regressors in the future dataframe
future["regressor_1"] = df["regressor_1"].mean()  # Filling missing values with mean
future["regressor_2"] = future["regressor_2"].interpolate(
    method="linear"
)  # Interpolate missing values

# Make predictions
forecast = model.predict(future)

# Plot forecast
fig = model.plot(forecast)
