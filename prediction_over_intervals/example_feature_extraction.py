import pandas as pd
from features import count_events_in_intervals


df = pd.DataFrame({
    'timestamp': [1, 2, 3, 4, 5, 6],
    'event_type': ['A', 'B', 'A', 'C', 'B', 'A']
})

result = count_events_in_intervals(df, 'timestamp', 'event_type')
print(result)
