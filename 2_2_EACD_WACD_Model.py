import pandas as pd
import numpy as np
import datetime
import Process_LevelIII_Data
import Auto_ACD_Model
import Auto_WACD_Model
from statsmodels.tsa.stattools import adfuller


# DATA TREATMENT
# Read the LevelIII data
csco_data = pd.read_csv("csco_levelIII_data.csv")
dt = datetime.date(2025,7, 9)
# Convert Timestamp column into datetime object
csco_data["Timestamp"] = pd.to_datetime(csco_data["Timestamp"], unit="ms", origin=dt)
csco_data = csco_data.drop(columns=["Ticker", "MPID"])

# Process the LevelIII Data
order_book, trades = Process_LevelIII_Data.process_book_data(csco_data)
trades.set_index("Timestamp", inplace=True)

# Keep only trades during opening market
trades = trades.loc[
            datetime.datetime.combine(dt,datetime.time(9,30,0)):
            datetime.datetime.combine(dt,datetime.time(15,59,59))
         ]

# Process duration between trades
trades["Duration"] = trades.index.to_series().diff().dt.total_seconds()
trades = trades.dropna(subset=["Duration"])

# Resampling trade in 5-min intervals + apply for each group the process_trades function
five_min_agg = trades.groupby(pd.Grouper(level=0, freq="5min"))["Duration"].mean().reset_index().rename(columns={"Duration": "AvgDuration"})

# Merge 5-min agg and trades df
trades = trades.reset_index()
trades = pd.merge_asof(trades, five_min_agg[["Timestamp", "AvgDuration"]], on="Timestamp", direction="backward")
# Process adjusted duration column
trades["AdjDuration"] = trades["Duration"]/np.exp(trades["AvgDuration"])
trades.set_index("Timestamp", inplace=True)

# Test the stationarity of the time series
result = adfuller(trades["AdjDuration"])
print(f"ADF Statistic: {result[0]}, p-value: {result[1]}")

# Send x adjduration series to fit ACD and WACD model
x = trades["AdjDuration"].values
x = x[x != 0]
Auto_ACD_Model.select_best_acd_model(x, p_max=3, q_max=3)
Auto_WACD_Model.select_best_wacd_model(x, p_max=3, q_max=3)


