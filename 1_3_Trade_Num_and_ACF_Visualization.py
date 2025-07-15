import pandas as pd
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

# PLOT
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,16))
_ = five_min_agg.numTrades.plot(ax=ax1, marker="o")
_ = ax1.set_title("Number of Trades during Continuous Time", fontsize=16)
_ = ax1.set_xlabel("Time")
_ = ax1.set_ylabel("Trades")

_ = plot_acf(five_min_agg.numTrades, ax=ax2)
_ = ax2.set_xlabel("Lags")
_ = ax2.set_ylabel("ACF")
plt.show()

# Intraday period pattern with more transactions at the open and the close of the day



