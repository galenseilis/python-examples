import pandas as pd
from itertools import combinations_with_replacement
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Step 1: Generate toy dataset
df = pd.DataFrame({
    'timestamp': [1, 2, 3, 4, 5, 6],
    'event_type': ['A', 'B', 'A', 'C', 'B', 'A']
})

# Step 2: Define the interval-counting function
def count_events_in_intervals(df, time_col, event_col):
    df = df.sort_values(by=time_col)
    time_points = df[time_col].unique()
    event_types = df[event_col].unique()
    intervals = list(combinations_with_replacement(time_points, 2))
    rows = []
    for start, end in intervals:
        mask = (df[time_col] >= start) & (df[time_col] <= end)
        subset = df[mask]
        counts = subset[event_col].value_counts().reindex(event_types, fill_value=0)
        row = {'start': start, 'end': end}
        row.update(counts.to_dict())
        rows.append(row)
    return pd.DataFrame(rows)

# Step 3: Generate training data from intervals
interval_df = count_events_in_intervals(df, 'timestamp', 'event_type')

# Step 4: Feature engineering
interval_df['duration'] = interval_df['end'] - interval_df['start'] + 1  # Include both endpoints

# Define features (start, end, duration) and targets (event counts)
X = interval_df[['start', 'end', 'duration']]
y = interval_df[['A', 'B', 'C']]

# Step 5: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 6: Train model
model = MultiOutputRegressor(RandomForestRegressor(random_state=42))
model.fit(X_train, y_train)

# Step 7: Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')

# Show results
for i, event in enumerate(y.columns):
    print(f"Event '{event}': MSE = {mse[i]:.2f}")

# Optional: predict on a new interval
example = pd.DataFrame({'start': [2], 'end': [5]})
example['duration'] = example['end'] - example['start'] + 1
prediction = model.predict(example)
print("\nPrediction for interval [2, 5]:", dict(zip(y.columns, prediction[0])))
