import bs4 as bs
import datetime as dt
import random
import os
import pandas_datareader.data as web
import pandas as pd
import pickle
import requests
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data as pdr
from pandas.core.common import flatten

df = pd.read_csv("sp500_joined_close_price.csv", parse_dates = ["Date"], index_col = 0)

random.seed(123)
log_ret = np.log(df/df.shift(1))

# this is the variance, used in the for loop to calculate the volatility
vol = log_ret.std()
# init. empty data frame
results = pd.DataFrame(columns = ["Size", "Return", "Volatility"])
for i in range(100000):
    '''
    In this function we use random module to generate random integers (and list of integers)
    to use along side the pandas .iloc method to get the right subset of randomly generated
    portfolios
    '''
    # random size of portfolio between 1 and 100
    randomsize = random.randint(1, 100)
    # using randomsize to get same amount of random values between 1 and 500
    assets = random.sample(range(0,500), randomsize)
    # picking a random day to sell the position
    randomday = random.sample(range(0,2556), 1)
    # saving the size since that is one of our measures of volatility, i.e more assets = less vol.
    size = len(assets)
    # Calculating the mean return for said portfolio on said day
    mean = log_ret.iloc[randomday, assets].mean(axis = 1)
    # Calculating the std(volatility) for random portfolio 
    std = np.sqrt(vol.iloc[assets].sum())/len(assets)
    #Append the results to the empty data frame
    results = results.append({"Size": size, "Return": mean[0], "Volatility": std}, ignore_index = True)
# df can be used for further ploting or calculation
df = results.sort_values(by = "Size", ascending = False)
# This is the answer to part 1, use for plotting and to generate describtive statistics
df2 = df.groupby('Size').mean()