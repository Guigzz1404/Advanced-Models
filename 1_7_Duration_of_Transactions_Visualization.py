import pandas as pd
import datetime
import Process_LevelIII_Data
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates


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

fig, ax = plt.subplots(figsize=(12,8))
_ = trades["Duration"].plot(ax=ax)
_ = ax.set_title("Duration of Transactions", fontsize=16)
_ = ax.set_xlabel("Time")
_ = ax.set_ylabel("Seconds")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
sns.despine()
plt.tight_layout()
plt.show()




