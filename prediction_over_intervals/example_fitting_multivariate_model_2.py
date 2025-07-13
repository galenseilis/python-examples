import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error

# --- Step 1: Sample time series event data ---
df = pd.DataFrame({
    'timestamp': [1, 2, 3, 4, 5, 6, 7, 9],
    'event_type': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A']
})

# --- Step 2: Function to count events in closed intervals ---
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

# --- Step 3: Create interval dataset ---
interval_df = count_events_in_intervals(df, 'timestamp', 'event_type')
interval_df['duration'] = interval_df['end'] - interval_df['start'] + 1

# Features and targets
X = interval_df[['start', 'end', 'duration']].values
y = interval_df[['A', 'B', 'C']].values
event_types = ['A', 'B', 'C']

# --- Step 4: Train multi-output model ---
base_model = HistGradientBoostingRegressor()
model = MultiOutputRegressor(base_model)
model.fit(X, y)

# --- Step 5: Predict on historical + future intervals ---
max_time = df['timestamp'].max()
future_times = [max_time + 1, max_time + 2]
all_times = np.sort(np.concatenate([df['timestamp'].unique(), future_times]))

# Predict counts at each single time point (interval [t, t])
interval_features = pd.DataFrame({'start': all_times, 'end': all_times})
interval_features['duration'] = 1
X_pred = interval_features[['start', 'end', 'duration']].values
preds_all = model.predict(X_pred)

# --- Step 6: Actual event counts at each time ---
true_counts = (
    df.groupby(['timestamp', 'event_type']).size()
    .unstack(fill_value=0)
    .reindex(all_times, fill_value=0)
)

# --- Step 7: Plot predictions vs true data ---
fig, axes = plt.subplots(len(event_types), 1, figsize=(10, 6), sharex=True)

for i, event in enumerate(event_types):
    axes[i].plot(all_times, preds_all[:, i], label='Model prediction', linestyle='--', color='orange')
    axes[i].plot(true_counts.index, true_counts[event], label='True count', marker='o', color='blue')
    axes[i].set_title(f"Event '{event}'")
    axes[i].legend()
    axes[i].set_ylabel('Count')

axes[-1].set_xlabel('Time')
plt.suptitle("Event Count Predictions with HistGradientBoostingRegressor")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
