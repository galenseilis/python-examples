import matplotlib.pyplot as plt
import pandas as pd

pd.set_option('display.max_rows', 10000)

df = pd.read_csv('trace_log.txt')

#df['Timestamp'].diff().dropna().hist(bins=100)
#plt.yscale('log')
#plt.ylabel('Frequency')
#plt.xlabel('Step Duration (seconds)')
#plt.show()

#print(df['Function'].value_counts())

#print(df.Timestamp.describe())
df.Timestamp.plot()
plt.show()
