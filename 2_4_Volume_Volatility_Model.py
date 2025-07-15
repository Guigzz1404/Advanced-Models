import pandas as pd
import datetime
import Process_LevelIII_Data
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns


def process_trades(df):

    num_trades = df.shape[0] # Number of lines in the df
    if num_trades > 0:
        avg_price = df["Price"].mean() # We take the mean price of the df
    else:
        avg_price = 0.0
    vol = df["Quantity"].sum()
    return pd.Series({"numTrades": num_trades, "avgPrice": avg_price, "volume": vol})


# Two equations to calculate It, we want parameters which minimize the "loss"
def clark_objective(params, r, V, Z1, Z2):
    sigma, mu_v, sigma2 = params
    if sigma <= 0 or mu_v <= 0 or sigma2 <= 0:
        return np.inf

    # I_t estimated from return
    I_r = (r / (sigma * Z1)) ** 2
    # I_t estimated from volume
    # We approximate sqrt(I_t) = abs(r) / (sigma * abs(Z1)) (I_t from return)
    sqrt_I = np.abs(r) / (sigma * np.abs(Z1))
    I_v = (V - sigma2 * sqrt_I * Z2) / mu_v

    # Quadratic error between two estimations of It
    loss = np.mean((I_r - I_v) ** 2)
    return loss


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

# Process returns and volatility columns
five_min_agg["returns"] = np.log(five_min_agg["avgPrice"]).diff()
five_min_agg["volatility"] = five_min_agg["returns"] ** 2

# CLARK AND TAUCHEN-PITTS MODEL
# Z1 and Z2 generation (standard normal variables)
np.random.seed(100) # We generate the same random standard normal variables (seed 14) at each execution
five_min_agg["Z1"] = np.random.normal(0, 1, size=len(five_min_agg))
five_min_agg["Z2"] = np.random.normal(0, 1, size=len(five_min_agg))


# Init parameters
initial_guess = [0.01, 1000, 1.0]
bounds = [(1e-6, None), (1e-3, None), (1e-6, None)]

res = minimize(clark_objective, initial_guess,
               args=(five_min_agg["returns"], five_min_agg["volume"], five_min_agg["Z1"], five_min_agg["Z2"]),
               bounds=bounds)

sigma_hat, mu_v_hat, sigma2_hat = res.x
print("sigma_hat:", sigma_hat)
print("mu_v_hat:", mu_v_hat)
print("sigma2_hat:", sigma2_hat)

# Calcul of I_t according to sigma value
five_min_agg["I_hat"] = (five_min_agg["returns"] / (sigma_hat * five_min_agg["Z1"])) ** 2


# Visualization of I_t
fig1, ax1 = plt.subplots(figsize=(12,8))
_ = five_min_agg["I_hat"].plot(ax=ax1, title="I_t", marker="o")
_ = ax1.set_xlabel("Time")
sns.despine()
plt.tight_layout()
plt.show()
