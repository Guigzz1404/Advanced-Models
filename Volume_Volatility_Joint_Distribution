import pandas as pd
import numpy as np
import datetime
import Process_LevelIII_Data
import seaborn as sns
import matplotlib.pyplot as plt


def process_trades(df):

    num_trades = df.shape[0] # Number of lines in the df
    if num_trades > 0:
        avg_price = df["Price"].mean() # We take the mean price of the df
    else:
        avg_price = 0.0
    vol = df["Quantity"].sum()
    return pd.Series({"numTrades": num_trades, "avgPrice": avg_price, "volume": vol})


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

five_min_agg["returns"] = np.log(five_min_agg["avgPrice"]).diff()
five_min_agg["volatility"] = five_min_agg["returns"] ** 2
print(five_min_agg)

# Visualization of joint distribution between Volume and Volatility
sns.jointplot(data=five_min_agg, x="volume", y="volatility", kind='kde').fig.set_size_inches(12, 8)
sns.despine()
plt.tight_layout()
plt.show()

# Little positive relation between volume and volatility
