#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 23 22:44:24 2018

@author: chongjiayoong
"""
import os, quandl, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# download data from Quandl website 
def data_arrays(ID):
    try:
        data = open('{}.pkl'.format(ID).replace('/','-'), 'rb')
        df = pickle.load(data)   
    except (OSError, IOError) as e:
        df = quandl.get(ID, returns="pandas")
        df.to_pickle(data)
    return df

def AC(x):
    x = np.array(x)
    arr = x - sum(x)/(len(x))
    auto = np.correlate(arr, arr, mode = 'full') 
    auto = auto[len(auto)//2:] 
    auto /= auto[::-1][-1]
    return auto
 
btc = data_arrays('BCHARTS/COINBASEUSD')
eth = data_arrays('GDAX/ETH_EUR')
ltc = data_arrays('GDAX/LTC_EUR')

# Creating an average value form high and low
ltc['Mean'] = (ltc['High']+ltc['Low'])*(1/2)
eth['Mean'] = (eth['High']+eth['Low'])*(1/2)
btc['Mean'] = (btc['High']+btc['Low'])*(1/2)

# Setting up dates
eth = eth['2017-2-20':'2018-4-4']
ltc = ltc['2017-2-20':'2018-4-4']
btc = btc['2017-2-20':'2018-4-4']

# Calculate autocorrelation and transform result to Pandas Dataframe
eth = AC(eth['Mean'])

#eth = eth/max(eth)

btc = AC(btc['Mean'])

#btc = btc/max(btc)

ltc = AC(ltc['Mean'])

#ltc = ltc/max(ltc)

#eth_vs_btc = np.correlate(eth,btc,'full')
#num = len(eth_vs_btc)
#t = np.arange(-num/2,num/2)
#plt.plot(t, eth_vs_btc) 
#plt.title("Cross-correlation (ETH vs BTC)")

#eth_vs_ltc = np.correlate(eth,ltc,'full')
#num = len(eth_vs_ltc)
#t = np.arange(-num/2,num/2)
#plt.plot(t, eth_vs_ltc) 
#plt.title("Cross-correlation (ETH vs LTC)")

#btc_vs_ltc = np.correlate(btc,ltc,'full')
#num = len(btc_vs_ltc)
#t = np.arange(-num/2,num/2)
#plt.plot(t, btc_vs_ltc) 
#plt.title("Cross-correlation (BTC vs LTC)")

#plt.plot(btc) 
#plt.title("Bitcoin Price")
#plt.ylabel("USD")
#plt.xlabel("Time")

#plt.plot(ltc)
#plt.title("Litecoin Price")
#plt.ylabel("USD")
#plt.xlabel("Time")

#plt.plot(eth) 
#plt.title("Ethereum Price")
#plt.ylabel("USD")
#plt.xlabel("Time")

#plt.plot(eth, color= "black")
#plt.plot(ltc, color = "orange")
#plt.plot(btc, color ="gold") #normalized coeeficient for cross correlation
##plt.title("Plot for Auto-correlation for Litecoin") # Bitcoin, Ethereum too 
#plt.title("Plot for Cross-Correlation for Litecoin") # Bitcoin, Ethereum too 
#plt.ylabel("Auto-correlation")
#plt.xlabel("Date (Interval by week)")
#plt.legend(['ETH','LTC','XBT'],loc='upper right')
#plt.show() 










