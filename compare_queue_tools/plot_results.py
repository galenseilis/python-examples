import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("results.csv")


# Histogram
for group, groupdf in df.groupby("Command being timed"):
    plt.hist(groupdf["User time (seconds)"], label=group, bins=20)

plt.ylabel("Frequency")
plt.xlabel("Run Time (seconds)")
plt.xscale("log")
plt.legend()
plt.savefig("run_time_hist.png", dpi=300, transparent=True)
plt.close()


# Markdown table
result = ""

for group, group_df in df.groupby("Command being timed"):
    result += group.replace('"', "`") + "\n\n"
    result += group_df.describe().to_markdown() + "\n\n"

with open("results.md", "w") as f:
    f.write(result)
