import pandas as pd
from itertools import combinations_with_replacement

def count_events_in_intervals(df, time_col, event_col):
    """
    For each pair of time points (start, end), counts the number of events
    of each type that occur in the closed interval [start, end].

    Parameters:
        df (pd.DataFrame): Input DataFrame
        time_col (str): Name of the time column
        event_col (str): Name of the event type column

    Returns:
        pd.DataFrame: A new DataFrame with 'start', 'end', and event count columns
    """
    # Ensure time is sorted
    df = df.sort_values(by=time_col)

    # Get unique time points and event types
    time_points = df[time_col].unique()
    event_types = df[event_col].unique()

    # Prepare all (start, end) intervals with start <= end
    intervals = list(combinations_with_replacement(time_points, 2))

    # Collect rows
    rows = []
    for start, end in intervals:
        mask = (df[time_col] >= start) & (df[time_col] <= end)
        subset = df[mask]
        counts = subset[event_col].value_counts().reindex(event_types, fill_value=0)
        
        row = {'start': start, 'end': end}
        row.update(counts.to_dict())
        rows.append(row)

    # Return as DataFrame
    return pd.DataFrame(rows)
