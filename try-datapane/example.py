import altair as alt
import datapane as dp
import pandas as pd
from vega_datasets import data

df = data.iris()
columns = list(df.columns)
print(columns)

view = dp.DataTable(df, label="Data")

dp.save_report(view, "quickstart_report.html", open=True)
