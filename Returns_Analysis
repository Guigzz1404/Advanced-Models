import pandas as pd
import numpy as np
import datetime
import Process_LevelIII_Data
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
from pandas.plotting import register_matplotlib_converters


# Display settings
sns.set_theme(style="darkgrid")
register_matplotlib_converters()
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16


def process_trades(df):

    num_trades = df.shape[0] # Number of lines in the df
    if num_trades > 0:
        last_price = df.Price.values[-1] # We take the last value of the df
    else:
        last_price = 0.0
    return pd.Series({"numTrades": num_trades, "lastPrice": last_price})


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

returns = np.log(five_min_agg["lastPrice"]).diff().dropna()

# PLOT
fig1, ax1 = plt.subplots(figsize=(12,8))
_ = returns.plot(ax=ax1, title="Log-return series", marker="o")
_ = ax1.set_xlabel("Time")
fig2, ax2 = plt.subplots(figsize=(12,8))
_ = plot_acf(returns, ax=ax2)
_ = ax2.set_xlabel("Lags")
_ = ax2.set_ylabel("ACF")
plt.show()
