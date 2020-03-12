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
names = []
for i in range(10):
    '''
    In this function we use random module to generate random integers (and list of integers)
    to use along side the pandas .iloc method to get the right subset of randomly generated
    portfolios
    '''
    weights = np.random.random(20)  
    weights /= np.sum(weights)
    # size of portfolio is now fixed at 20
    size = 20
    population = range(500)
    # using randomsize to get same amount of random values between 1 and 500
    assets = random.sample(population, size)
    # picking a random day to sell the position
    a = log_ret.columns.values
    names.append(a)
    randomday = random.sample(range(0,2556), 1)
    # Calculating the mean return for said portfolio on said day
    mean = log_ret.iloc[randomday, assets].mean(axis = 1)
    # Calculating the std(volatility) for random portfolio 
    std = np.sqrt(vol.iloc[assets].sum())/len(assets)
    #Append the results to the empty data frame
    results = results.append({"Size": size, "Return": mean[0], "Volatility": std}, ignore_index = True)

print(len(names[1]))