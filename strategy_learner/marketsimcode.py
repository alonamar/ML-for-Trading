# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:35:33 2018

Student Name: Alon Amar (replace with your name)
GT User ID: aamar32 (replace with your User ID)
GT ID: 903339940 (replace with your GT ID)
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data

def author():
    return 'aamar32'

def compute_portvals(ordersDF, start_val = 1000000, commission=0.0, impact=0.0):
    ordersDF.sort_index(inplace=True)
    symbols = ordersDF['Symbol'].unique().tolist()
    start_date = ordersDF.index.min()
    end_date = ordersDF.index.max()
    pricesDF = get_data(symbols, pd.date_range(start_date, end_date)).drop(['SPY'], axis=1)
    pricesDF['CASH'] = 1
    tradesDF = pd.DataFrame(0,index=pricesDF.index, columns=pricesDF.columns)
    for index, row in ordersDF.iterrows():
        if index in tradesDF.index:
            if row["Order"] == 'BUY':
                tradesDF.loc[index,row["Symbol"]] += row["Shares"]
                tradesDF.loc[index,'CASH'] += -row["Shares"] * (1 + impact) * pricesDF.loc[index,row["Symbol"]] - commission
            elif row["Order"] == 'SELL':
                tradesDF.loc[index,row["Symbol"]] -= row["Shares"]
                tradesDF.loc[index,'CASH'] += row["Shares"] * (1 - impact) * pricesDF.loc[index,row["Symbol"]] - commission
            else:
                pass
    holdingsDF = tradesDF.copy(deep=True)
    holdingsDF.iloc[0, len(holdingsDF.columns)-1] += start_val
    holdingsDF = holdingsDF.cumsum()
    valueDF = pricesDF * holdingsDF
    portvals = pd.DataFrame(valueDF.sum(axis=1), columns=['SUM'])
    return portvals

def calc_stats(daily_val):
    daily_ret = (daily_val / daily_val.shift(1)) - 1
    daily_ret = daily_ret[1:]
    cr = (daily_val[-1]/daily_val[0]) - 1
    adr = daily_ret.mean()
    sddr = daily_ret.std()
    k = np.sqrt(252)
    if sddr != 0:
        sr = k * (adr/sddr)
    else:
        sr=0
    return cr, adr, sddr, sr

### tranformOrders change the output of testPolicy to a format of the Marketsim assignment 
def tranformOrders(df, ordersDF=None):
    symDF = pd.DataFrame(0, index=df.index, columns=['Symbol', 'Order', 'Shares'])
    symDF['Shares'] = df.copy()
    symDF.loc[symDF['Shares'] > 0, 'Order'] = "BUY"
    symDF.loc[symDF['Shares'] < 0, 'Order'] = "SELL"
    symDF.loc[symDF['Shares'] < 0, 'Shares'] *= -1
    #symDF = symDF[symDF['Order'] != 0]
    symDF['Symbol'] = df.columns[0]
    if isinstance(ordersDF, pd.DataFrame):
        symDF = pd.concat([symDF,ordersDF])
    return symDF

### return DF with only one buy order of 1000 at start and holding
def benchmark(symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
    pricesDF = get_data([symbol], pd.date_range(sd, ed)).drop(['SPY'], axis=1)
    myDF = pricesDF.copy()
    myDF.iloc[:,:] = 0
    myDF.iloc[0] = 1000
    return compute_portvals(tranformOrders(myDF), sv)


if __name__ == "__main__":  		   	  			    		  		  		    	 		 		   		 		  
    pass


