import subprocess
import time

import matplotlib.pyplot as plt
import pandas as pd

REPLICATES = 1000

rust_cmd = ["./rust_mm1"]
ciw_cmd = ["python", "ciw_MM1.py"]
pypy_ciw_cmd = ["pypy3", "ciw_MM1.py"]
simpy_cmd = ["python", "simpy_MM1.py"]
pypy_simpy_cmd = ["pypy3", "simpy_MM1.py"]
qt_cmd = ["python", "qt_MM1.py"]
cpp_cmd = ["./cpp_MM1"]

COMMANDS = [rust_cmd, cpp_cmd, ciw_cmd, pypy_ciw_cmd, simpy_cmd, pypy_simpy_cmd, qt_cmd]


def capture_execution_time(command):
    try:
        # Run the command and capture both stdout and stderr
        result = subprocess.run(
            ["/usr/bin/time", "-v"] + command,
            stderr=subprocess.PIPE,
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )

        # Check for errors
        result.check_returncode()

        # Extract time information from stderr
        time_info = result.stderr

        return time_info

    except subprocess.CalledProcessError as e:
        # Handle errors, if any
        print(f"Error: {e}")
        return None


results = []
for command in COMMANDS:
    for _ in range(REPLICATES):
        result = capture_execution_time(command)
        if result is not None:
            result = [
                line.replace("\t", "").replace("\n", "").split(": ")
                for line in result.split("\n\t")
            ]
            result = {i[0]: i[1] for i in result}
            print(_, result["Command being timed"], result["User time (seconds)"])
            results.append(result)


df = pd.DataFrame(results)

for col in df.columns:
    try:
        df[col] = df[col].astype(float)
    except Exception as e:
        print(e)

df.to_csv("results.csv", index=False)
