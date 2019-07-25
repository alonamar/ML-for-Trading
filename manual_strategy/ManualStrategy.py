# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 16:42:46 2018

@author: Alon
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt
import indicators as ind
import marketsimcode as mrk



def testPolicy(symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
    # get volume and prices
        pricesDF = get_data([symbol], pd.date_range(sd, ed)).drop(['SPY'], axis=1)
        volumeDF = get_data([symbol], pd.date_range(sd, ed), colname="Volume").drop(['SPY'], axis=1)
        closeDF = get_data([symbol], pd.date_range(sd, ed), colname="Close").drop(['SPY'], axis=1)
        volumeDF = (volumeDF * closeDF / pricesDF)
        
        # calc indicators
        bbpDF = ind.bbp(pricesDF, 10)
        rsi_obvDF = ind.rsi_obv(pricesDF, volumeDF, 14)
        #trixDF = ind.trix(pricesDF, 14)
        
        # check conditions:
        # close when trix crosses the avarage
        # buy - when bbf < 0 or rsi of obv < 30
        # sell - when bbf > 1 or rsi of obv > 70
        '''trix_cross = pd.DataFrame(0, index=trixDF.index, columns=["JPM"])
        trix_cross[trixDF >= 1] = 1
        trix_cross[1:] = trix_cross.diff()'''
        orders = pd.DataFrame(np.nan, index=pricesDF.index, columns=pricesDF.columns)
        orders[(bbpDF < 0) | (rsi_obvDF < 30)] = 1000
        orders[(bbpDF > 1) | (rsi_obvDF > 70)] = -1000
        #orders[(trix_cross != 0)] = 0
        orders.ffill(inplace=True)
        orders.fillna(0, inplace=True)
        orders = orders.diff()
        orders.iloc[0] = 0
        return orders



def test_code():
    
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    orders = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    jpmDF = mrk.tranformOrders(orders)
    jpmDF = mrk.compute_portvals(jpmDF, 100000, commission=9.95, impact=0.005)
    norm_JpmDF = jpmDF / jpmDF.iloc[0]
    
    bench_jpmDF = mrk.benchmark("JPM",  sd=sd, ed=ed, sv=100000)
    norm_bench_jpmDF = bench_jpmDF / bench_jpmDF.iloc[0]
    
    plt.figure(figsize=(14, 6))
    plt.plot(norm_JpmDF, 'black', label = "Manual")
    plt.plot(norm_bench_jpmDF, 'blue', label = "Benchmark")
    
    buyLines = orders[orders['JPM'] > 0].index
    sellLines = orders[orders['JPM'] < 0].index
    for x in buyLines:
        plt.axvline(x=x, color='green')
    for x in sellLines:
        plt.axvline(x=x, color='red')
    #priceDF = get_data(['JPM'], pd.date_range(sd, ed)).drop(['SPY'], axis=1)
    #normPrice = priceDF/priceDF.iloc[0]
    #plt.plot(normPrice)
    plt.legend()
    plt.savefig("Manual.png", dpi=300)

    port_cr, port_adr, port_sddr, port_sr = mrk.calc_stats(jpmDF['SUM'])
    bench_cr, bench_adr, bench_sddr, bench_sr = mrk.calc_stats(bench_jpmDF['SUM'])
    # Compare portfolio against $SPX
    #print "Date Range: {} to {}".format(start_date, end_date)
    print 
    print "Sharpe Ratio of Manual: {}".format(port_sr)
    print "Sharpe Ratio of Benchmark: {}".format(bench_sr)
    print
    print "Cumulative Return of Manual: {}".format(port_cr)
    print "Cumulative Return of Benchmark: {}".format(bench_cr)
    print
    print "Standard Deviation of Manual: {}".format(port_sddr)
    print "Standard Deviation of Benchmark: {}".format(bench_sddr)
    print
    print "Average Daily Return of Manual: {}".format(port_adr)
    print "Average Daily Return of Benchmark: {}".format(bench_adr)


if __name__ == "__main__":  		   	  			    		  		  		    	 		 		   		 		  
    test_code()  	


