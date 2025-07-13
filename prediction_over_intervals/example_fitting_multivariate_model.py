import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations_with_replacement
from sklearn.cross_decomposition import PLSRegression
from features import count_events_in_intervals

# --- Step 1: Sample time series event data ---
df = pd.DataFrame({
    'timestamp': [1, 2, 3, 4, 5, 6, 7, 9],
    'event_type': ['A', 'B', 'A', 'C', 'B', 'A', 'C', 'A']
})

interval_df = count_events_in_intervals(df, 'timestamp', 'event_type')
interval_df['duration'] = interval_df['end'] - interval_df['start'] + 1

# --- Step 3: Train multivariate regression model ---
X = interval_df[['start', 'end', 'duration']].values
y = interval_df[['A', 'B', 'C']].values

model = PLSRegression(n_components=2)
model.fit(X, y)

# --- Step 4: Predict on full historical intervals + future ones ---
max_time = df['timestamp'].max()
future_times = [max_time + 1, max_time + 2]

# Create historical and future intervals of form [t, t]
time_points = df['timestamp'].unique()
all_points = np.concatenate([time_points, future_times])
all_points = np.sort(all_points)

intervals = [(t, t) for t in all_points]  # count at each time point

interval_features = pd.DataFrame(intervals, columns=['start', 'end'])
interval_features['duration'] = 1

X_all = interval_features[['start', 'end', 'duration']].values
preds_all = model.predict(X_all)

# --- Step 5: Actual counts at each timestamp ---
event_types = ['A', 'B', 'C']
true_counts = (
    df.groupby(['timestamp', 'event_type']).size()
    .unstack(fill_value=0)
    .reindex(all_points, fill_value=0)
)

# --- Step 6: Plot ---
fig, axes = plt.subplots(len(event_types), 1, figsize=(10, 6), sharex=True)

for i, event in enumerate(event_types):
    axes[i].plot(all_points, preds_all[:, i], label='Model prediction', linestyle='--', color='orange')
    axes[i].plot(true_counts.index, true_counts[event], label='True count', marker='o', color='blue')
    axes[i].set_title(f"Event '{event}'")
    axes[i].legend()
    axes[i].set_ylabel('Count')

axes[-1].set_xlabel('Time')

plt.suptitle("Event Counts: Historical vs. Predicted + Extrapolation")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
