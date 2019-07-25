# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:43:19 2018

@author: Alon
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt


def rsi(df, lookback=14):
    '''
    RSI > 70 (sell)
    RSI < 30 (buy)
    '''
    daily_rets = pd.DataFrame(df.diff())
    up_rets = daily_rets[daily_rets >= 0].fillna(0).cumsum()
    down_rets = -1 * daily_rets[daily_rets < 0].fillna(0).cumsum()
    up_gain = up_rets - up_rets.shift(lookback)
    down_loss = down_rets - down_rets.shift(lookback)
    rs = (up_gain/lookback)/(down_loss/lookback)
    rsi = 100 - (100/(1+rs))
    rsi[rsi==np.inf] = 100
    return rsi

def bbp(df, lookback=14):
    '''
    Bollinger Band % > 1 (sell)
    Bollinger Band % < 0 (buy)
    '''
    sma = pd.DataFrame(df.rolling(window=lookback, min_periods=lookback).mean())
    sma_std = pd.DataFrame(df.rolling(window=lookback, min_periods=lookback).std())
    top_band = sma + 2*sma_std
    bottom_band = sma - 2*sma_std
    bbp = (df - bottom_band) / (top_band - bottom_band)
    return bbp

def trix(pricesDF, lookback=14):
    '''
    TRIX crosses above the zero line it gives a buy signal
    When it closes below the zero line, it gives a sell signal.
    '''
    ema1 = pricesDF.ewm(span=lookback, min_periods=lookback).mean()
    ema2 = ema1.ewm(span=lookback, min_periods=lookback).mean()
    ema3 = ema2.ewm(span=lookback, min_periods=lookback).mean()
    trix = (ema3/ema3.shift(1)) - 1
    
    return trix

def obv(pricesDF, volumeDF):
    '''
    rising OBV reflects positive volume pressure that can lead to higher prices (buy)
    OBV reflects negative volume pressure that can foreshadow lower prices (sell)
    '''
    obvDF = pd.DataFrame(0, index=volumeDF.index, columns=volumeDF.columns)
    obvDF[pricesDF.diff(1) > 0] = volumeDF[pricesDF.diff(1) > 0]
    obvDF[pricesDF.diff(1) < 0] = -volumeDF[pricesDF.diff(1) < 0]
    obvDF = obvDF.cumsum()
    return obvDF

def rsi_obv(pricesDF, volumeDF, lookback=14):
    '''
    rsi_obv > 70 (sell)
    rsi_obv < 30 (buy)
    '''
    return rsi(obv(pricesDF, volumeDF), lookback)


def plot_helper(title, pricesDF, indDF, max_val, min_val, obv=False, cross=False, printLines=True, saveFig=False):
    plt.figure(figsize=(14, 6))
    subNum = [211, 212]
    if obv:
        subNum = [311, 312, 313]
    
    plt.subplot(subNum[0])
    plt.title(title)
    plt.grid(True)
    plt.plot((pricesDF), label = "Price", color='red')
    plt.legend()

    if obv:
        plt.subplot(subNum[1])
        plt.grid(True)
        plt.plot(indDF.fillna(0), label = "OBV")
        plt.legend()
        indDF = rsi(indDF)
        title = 'RSI_OBV'

    plt.subplot(subNum[-1])
    
    if cross:
        crossDF = pd.DataFrame(0, index=indDF.index, columns=["JPM"])
        crossDF[indDF >= 0] = 1
        crossDF[1:] = crossDF.diff()
        sellCoords = indDF[crossDF == -1]
        buyCoords = indDF[crossDF == 1]
    else:
        sellCoords = indDF[indDF > max_val]
        buyCoords = indDF[indDF < min_val]
    
    plt.grid(True)
    plt.plot(indDF.fillna(0), label = title)
    if printLines:
        plt.axhline(y=max_val, linestyle="--", color='black')
        plt.axhline(y=min_val, linestyle="--", color='black')
    plt.plot(sellCoords, marker='o', linestyle=' ', label = "SELL")
    plt.plot(buyCoords, marker='o', linestyle=' ', label = "BUY")
    plt.legend(loc=3)
    plt.tight_layout()
    if saveFig:
        plt.savefig(title + ".png", dpi=300)

#def plot_min_max(title, pricesDF, indDF, max_val, min_val, cross=False, printLines=True, saveFig=False):


def plot_indicators():
    start_date = dt.datetime(2008,1,1)
    end_date = dt.datetime(2009,12,31)
    pricesDF = get_data(['JPM'], pd.date_range(start_date, end_date)).drop(['SPY'], axis=1)
    volumeDF = get_data(['JPM'], pd.date_range(start_date, end_date), colname="Volume").drop(['SPY'], axis=1)
    closeDF = get_data(['JPM'], pd.date_range(start_date, end_date), colname="Close").drop(['SPY'], axis=1)
    volumeDF = (volumeDF * closeDF / pricesDF)
    
    
    trixDF = trix(pricesDF)
    rsiDF = rsi(pricesDF)
    obvDF = obv(pricesDF, volumeDF)
    bbpDF = bbp(pricesDF)
    
    saveFig = True
    plot_helper('BBP', pricesDF, bbpDF, 1, 0, saveFig=saveFig)
    plot_helper('RSI', pricesDF, rsiDF, 70, 30, saveFig=saveFig)
    plot_helper('TRIX', pricesDF, trixDF, 70, 30, cross=True, printLines=False, saveFig=saveFig)
    plot_helper('OBV', pricesDF, obvDF, 70, 30, obv=True, saveFig=saveFig)

    #for xc in xcoords:
        #plt.axvline(x=xc, color='black')
    #plt.annotate('local max', x=xcoords[0], xytext=(3, 1.5),
            #arrowprops=dict(facecolor='black', shrink=0.05))
    
    
    #markers_on = bbpDF[bbpDF['JPM'] > 1]
    #plt.plot(xs, ys, '-gD', markevery=markers_on)
    
    #plot_indicator(pricesDF, 'JPM', obvDF)
    
if __name__ == "__main__":  		   	  			    		  		  		    	 		 		   		 		  
    plot_indicators() 




