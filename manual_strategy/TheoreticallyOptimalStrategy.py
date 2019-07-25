# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 13:32:39 2018

@author: Alon
"""

import pandas as pd
import numpy as np
import datetime as dt
from util import get_data, plot_data
import matplotlib.pyplot as plt
import marketsimcode as mrk

### testPolicy sell/buy depends on the next day
### In case of change of orice direction, it will sell/buy maximum shares (-/+ 1000 or 2000)
def testPolicy(symbol, sd=dt.datetime(2008,1,1), ed=dt.datetime(2009,12,31), sv = 100000):
    pricesDF = get_data([symbol], pd.date_range(sd, ed)).drop(['SPY'], axis=1)
    myDF = pricesDF.diff(1).fillna(0)
    myDF[pricesDF.diff(1) < 0] = 1000
    myDF[pricesDF.diff(1) > 0] = -1000
    myDF = myDF.diff(-1).fillna(0)
    return myDF


def test_code():
    
    sd = dt.datetime(2008,1,1)
    ed = dt.datetime(2009,12,31)
    
    jpmDF = testPolicy(symbol = "JPM", sd=sd, ed=ed, sv = 100000)
    jpmDF = mrk.tranformOrders(jpmDF)
    jpmDF = mrk.compute_portvals(jpmDF, 100000)
    norm_JpmDF = jpmDF / jpmDF.iloc[0]
    
    bench_jpmDF = mrk.benchmark("JPM", sv=100000, sd=sd, ed=ed)
    norm_bench_jpmDF = bench_jpmDF / bench_jpmDF.iloc[0]
    plt.figure(figsize=(14, 6))
    plt.plot(norm_JpmDF, 'black', label = "Optimal")
    plt.plot(norm_bench_jpmDF, 'blue', label = "Benchmark")
    plt.legend()
    plt.grid()
    plt.savefig("Optimal.png", dpi=300)

    port_cr, port_adr, port_sddr, port_sr = mrk.calc_stats(jpmDF['SUM'])
    bench_cr, bench_adr, bench_sddr, bench_sr = mrk.calc_stats(bench_jpmDF['SUM'])
    # Compare portfolio against $SPX
    #print "Date Range: {} to {}".format(start_date, end_date)
    print 
    print "Sharpe Ratio of Optimal: {}".format(port_sr)
    print "Sharpe Ratio of Benchmark: {}".format(bench_sr)
    print
    print "Cumulative Return of Optimal: {}".format(port_cr)
    print "Cumulative Return of Benchmark: {}".format(bench_cr)
    print
    print "Standard Deviation of Optimal: {}".format(port_sddr)
    print "Standard Deviation of Benchmark: {}".format(bench_sddr)
    print
    print "Average Daily Return of Optimal: {}".format(port_adr)
    print "Average Daily Return of Benchmark: {}".format(bench_adr)


if __name__ == "__main__":  		   	  			    		  		  		    	 		 		   		 		  
    test_code()  	