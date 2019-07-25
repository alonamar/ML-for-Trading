# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 21:43:19 2018

Student Name: Alon Amar (replace with your name)
GT User ID: aamar32 (replace with your User ID)
GT ID: 903339940 (replace with your GT ID)
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data
import matplotlib.pyplot as plt


def author(self):
    return 'aamar32'

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




