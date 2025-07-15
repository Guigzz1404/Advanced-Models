import pandas as pd
import datetime
import Process_LevelIII_Data
from arch import arch_model
import numpy as np


def garch_grid_search(series, p_range=(1,3), q_range=(1,3)):

    best_aic = np.inf # Infinity positive
    best_order = None
    best_model = None

    for p in range(p_range[0], p_range[1]+1):
        for q in range(q_range[0], q_range[1]+1):
            try:
                model = arch_model(series, vol="GARCH", p=p, q=q)
                res = model.fit(disp="off")
                if res.aic < best_aic:
                    best_aic = res.aic
                    best_order = (p,q)
                    best_model = res
            except Exception as e:
                continue  # skip models that fail

    return best_order, best_model


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
durations = trades.index.to_series().diff().dt.total_seconds().dropna()

# Fit the best GARCH model for the series
garch_order, garch_model = garch_grid_search(durations)
print("Best (p,q):", garch_order)
print(garch_model.summary())

