import pandas as pd
import re
import requests
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
from datetime import datetime

# Step 1: Fetch the PIP version and date data
url = "https://raw.githubusercontent.com/pypa/pip/main/NEWS.rst"
response = requests.get(url)
content = response.text

# Regular expression pattern to match the version number and date
pattern = re.compile(r"(\d+\.\d+(\.\d+)?[a-z]*\d*) \((\d{4}-\d{2}-\d{2})\)")

# Find all matches in the content
matches = pattern.findall(content)

# Extract only the necessary groups
matches = [(match[0], match[2]) for match in matches]

# Create a DataFrame from the matches
df = pd.DataFrame(matches, columns=["Version", "Date"])

# Convert date strings to datetime objects
df["Date"] = pd.to_datetime(df["Date"])

# Sort the DataFrame by date
df = df.sort_values(by="Date")

# Separate major, minor, and patch versions
df["Major"] = df["Version"].apply(lambda x: x.split(".")[0])
df["Minor"] = df["Version"].apply(
    lambda x: x.split(".")[1] if len(x.split(".")) > 1 else "0"
)
df["Patch"] = df["Version"].apply(
    lambda x: x.split(".")[2] if len(x.split(".")) > 2 else "0"
)

major_versions = df[df["Minor"] == "0"][df["Patch"] == "0"]
minor_versions = df[df["Minor"] != "0"][df["Patch"] == "0"]
patch_versions = df[df["Patch"] != "0"]


def plot_versions(dates, releases, title, filename):
    # Choose some nice levels: alternate minor releases between top and bottom, and
    # progressively shorten the stems for bugfix releases.
    levels = []
    major_minor_releases = sorted({release[:3] for release in releases})
    for release in releases:
        major_minor = release[:3]
        bugfix = int(release[4]) if len(release) > 4 and release[4].isdigit() else 0
        h = 1 + 0.8 * (5 - bugfix)
        level = h if major_minor_releases.index(major_minor) % 2 == 0 else -h
        levels.append(level)

    # The figure and the axes.
    fig, ax = plt.subplots(figsize=(8.8, 4), layout="constrained")
    ax.set(title=title)

    # The vertical stems.
    ax.vlines(
        dates,
        0,
        levels,
        color=[
            ("tab:red", 1 if release.endswith(".0") else 0.5) for release in releases
        ],
    )
    # The baseline.
    ax.axhline(0, c="black")
    # The markers on the baseline.
    minor_dates = [date for date, release in zip(dates, releases) if release[-1] == "0"]
    bugfix_dates = [
        date for date, release in zip(dates, releases) if release[-1] != "0"
    ]
    ax.plot(bugfix_dates, np.zeros_like(bugfix_dates), "ko", mfc="white")
    ax.plot(minor_dates, np.zeros_like(minor_dates), "ko", mfc="tab:red")

    # Annotate the lines.
    for date, level, release in zip(dates, levels, releases):
        ax.annotate(
            release,
            xy=(date, level),
            xytext=(-3, np.sign(level) * 3),
            textcoords="offset points",
            verticalalignment="bottom" if level > 0 else "top",
            weight="bold" if release.endswith(".0") else "normal",
            bbox=dict(boxstyle="square", pad=0, lw=0, fc=(1, 1, 1, 0.7)),
        )

    ax.yaxis.set(
        major_locator=mdates.YearLocator(), major_formatter=mdates.DateFormatter("%Y")
    )

    # Remove the y-axis and some spines.
    ax.yaxis.set_visible(False)
    ax.spines[["left", "top", "right"]].set_visible(False)

    ax.margins(y=0.1)
    # Save the plot as a PNG file with 300 dpi
    fig.savefig(filename, dpi=300)
    plt.close(fig)


# Plotting Major Versions
plot_versions(
    major_versions["Date"],
    major_versions["Version"],
    "PIP Major Release Dates",
    "pip_major_releases.png",
)

# Plotting Minor Versions
plot_versions(
    minor_versions["Date"],
    minor_versions["Version"],
    "PIP Minor Release Dates",
    "pip_minor_releases.png",
)

# Plotting Patch Versions
plot_versions(
    patch_versions["Date"],
    patch_versions["Version"],
    "PIP Patch Release Dates",
    "pip_patch_releases.png",
)
