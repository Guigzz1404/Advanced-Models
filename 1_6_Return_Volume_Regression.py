import pandas as pd
import numpy as np
import datetime
import Process_LevelIII_Data
import statsmodels.api as sm


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

five_min_agg['log_return'] = np.log(five_min_agg['lastPrice']) - np.log(five_min_agg['lastPrice'].shift(1))
five_min_agg = five_min_agg.dropna()

# Regression to see if a pattern exist between return and number of transactions
# Variables
X = five_min_agg["numTrades"]  # Explanatory Variable
Y = five_min_agg['log_return']  # Dependant Variable

# Add intercept (β0 constant)
X = sm.add_constant(X)

# Model
model = sm.OLS(Y, X).fit()

print(model.summary())

# The model can't say if there is a real correlation between return and number of trade bc P>|t| is not <0.05
# Plus β1 is close to 0

