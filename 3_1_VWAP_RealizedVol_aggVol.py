import pandas as pd
import datetime
import Process_LevelIII_Data
import numpy as np


def process_trades(df):

    num_trades = df.shape[0] # Number of lines in the df
    if num_trades > 0:
        # Average Price
        avg_price = df["Price"].mean()
        # VWAP
        vwap = (df["Price"]*df["Quantity"]).sum() / df["Quantity"].sum()
        # Realized Volatility
        realized_vol = ((np.log(df["Price"]).diff()) ** 2).sum()
        # Aggregated Volume
        vol = df["Quantity"].sum()

    else:
        avg_price = 0.0
        vwap = 0.0
        realized_vol = 0.0
        vol = 0.0
    return pd.Series({"avgPrice": avg_price, "vwap": vwap, "realizedVol": realized_vol, "aggVolume": vol})


# DATA TREATMENT
# Read the LevelIII data
csco_data = pd.read_csv("csco_levelIII_data.csv")
dt = datetime.date(2025,7, 9)
# Convert Timestamp column into datetime object
csco_data["Timestamp"] = pd.to_datetime(csco_data["Timestamp"], unit="ms", origin=dt)
csco_data = csco_data.drop(columns=["Ticker", "MPID"])

# Process the LevelIII Data
order_book, trades = Process_LevelIII_Data.process_book_data(csco_data)

# Resampling trade in 5-min intervals + apply for each group the process_trades function
five_min_agg = (trades.groupby(pd.Grouper(key="Timestamp", freq="5min")).apply(process_trades, include_groups=False))
five_min_agg = five_min_agg.loc[
                    datetime.datetime.combine(dt,datetime.time(9,30,0)):
                    datetime.datetime.combine(dt,datetime.time(15,59,59))
               ]

print(five_min_agg)