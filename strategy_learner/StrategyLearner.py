"""
Template for implementing StrategyLearner  (c) 2016 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332  		   	  			    		  		  		    	 		 		   		 		  
All Rights Reserved  		   	  			    		  		  		    	 		 		   		 		  

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students  		   	  			    		  		  		    	 		 		   		 		  
and other users of this template code are advised not to share it with others  		   	  			    		  		  		    	 		 		   		 		  
or to make it available on publicly viewable websites including repositories  		   	  			    		  		  		    	 		 		   		 		  
such as github and gitlab.  This copyright statement should not be removed  		   	  			    		  		  		    	 		 		   		 		  
or edited.  		   	  			    		  		  		    	 		 		   		 		  

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future  		   	  			    		  		  		    	 		 		   		 		  
students of CS 7646 is prohibited and subject to being investigated as a  		   	  			    		  		  		    	 		 		   		 		  
GT honor code violation.  		   	  			    		  		  		    	 		 		   		 		  

-----do not edit anything above this line---

Student Name: Alon Amar (replace with your name)
GT User ID: aamar32 (replace with your User ID)
GT ID: 903339940 (replace with your GT ID)
"""

import datetime as dt
import pandas as pd
import numpy as np
import util as ut
import indicators as ind
import QLearner as ql


class StrategyLearner(object):

    @staticmethod
    def author():
        return 'aamar32'

    # constructor
    def __init__(self, verbose=False, impact=0.0):
        self.verbose = verbose
        self.impact = impact
        self.steps = 80
        self.min_iter = 10
        self.hold_state = 0
        self.learner = ql.QLearner(num_states=self.steps * self.steps,  # long/short * ind1 * ind2
                                   num_actions=3,  # long/short/cash
                                   alpha=0.2,
                                   gamma=0.9,
                                   rar=0.98,
                                   radr=0.999,
                                   dyna=0,
                                   verbose=self.verbose)

        # this method should create a QLearner, and train it for trading

    def getState(self, numList):
        s = numList[0] * self.steps + numList[1]
        return s

    def discretizing(self, data):
        return pd.qcut(data, self.steps - 2, retbins=True, duplicates='drop')[1]

    def getDiscretValue(self, bins, val):
        return np.digitize(val, bins)

    def translateAction(self, a):
        if a == 0:
            return 0
        elif a == 1:
            return 1000
        elif a == 2:
            return -1000
        else:
            raise Exception('action should be either 0, 1 or 2. Action received: {}'.format(a))

    def calcImpact(self, hold):
        hold_value = self.hold_state
        self.hold_state = hold
        if hold_value == hold:
            return 1
        elif hold_value > hold:
            return 1 - self.impact
        else:
            return 1 + self.impact

    def addEvidence(self, symbol="IBM",
                    sd=dt.datetime(2008, 1, 1),
                    ed=dt.datetime(2009, 1, 1),
                    sv=10000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        pricesDF = prices_all[syms]
        prices_SPY = prices_all['SPY']
        volumeDF = ut.get_data(syms, dates, colname="Volume").drop(['SPY'], axis=1)
        closeDF = ut.get_data(syms, dates, colname="Close").drop(['SPY'], axis=1)
        volumeDF = (volumeDF * closeDF / pricesDF)
        if self.verbose:
            print pricesDF

        # calc indicators
        bbpDF = ind.bbp(pricesDF, 10)
        rsi_obvDF = ind.rsi_obv(pricesDF, volumeDF, 14)
        self.bbpBins = self.discretizing(bbpDF[symbol].values)
        self.obvBins = self.discretizing(rsi_obvDF[symbol].values)
        bbpDF = self.getDiscretValue(self.bbpBins, bbpDF[symbol].values)
        rsi_obvDF = self.getDiscretValue(self.obvBins, rsi_obvDF[symbol].values)

        prices = pricesDF[symbol].values
        orders = np.empty(len(prices))
        min_iter = 0
        while 1:
            old_orders = np.copy(orders)
            orders[0:15] = 0
            s = self.getState([bbpDF[14], rsi_obvDF[14]])
            self.learner.querysetstate(s)
            hold_value = 0
            impact_factor = 1
            for i in range(15, len(bbpDF)):
                r = ((prices[i] / (prices[i-1] * impact_factor)) - 1) * hold_value
                a = self.learner.query(s, r)
                hold_value = self.translateAction(a)
                orders[i] = hold_value
                impact_factor = self.calcImpact(hold_value)
                s = self.getState([bbpDF[i], rsi_obvDF[i]])
            if np.array_equal(orders, old_orders) and min_iter > self.min_iter: #converged and min iterations
                break
            min_iter += 1
        return orders

    def testPolicy(self, symbol="IBM",
                   sd=dt.datetime(2009, 1, 1),
                   ed=dt.datetime(2010, 1, 1),
                   sv=10000):
        syms = [symbol]
        dates = pd.date_range(sd, ed)
        prices_all = ut.get_data(syms, dates)
        pricesDF = prices_all[syms]
        prices_SPY = prices_all['SPY']
        volumeDF = ut.get_data(syms, dates, colname="Volume").drop(['SPY'], axis=1)
        closeDF = ut.get_data(syms, dates, colname="Close").drop(['SPY'], axis=1)
        volumeDF = (volumeDF * closeDF / pricesDF)
        if self.verbose:
            print pricesDF

        # calc indicators
        bbpDF = ind.bbp(pricesDF, 10)
        rsi_obvDF = ind.rsi_obv(pricesDF, volumeDF, 14)

        bbpDF = self.getDiscretValue(self.bbpBins, bbpDF[symbol].values)
        rsi_obvDF = self.getDiscretValue(self.obvBins, rsi_obvDF[symbol].values)
        prices = pricesDF[symbol].values
        orders = np.empty(len(prices))

        s = self.getState([bbpDF[14], rsi_obvDF[14]])
        self.learner.querysetstate(s)
        orders[0:15] = 0
        for i in range(15, len(bbpDF)):
            a = self.learner.querysetstate(s)
            orders[i] = self.translateAction(a)
            s = self.getState([bbpDF[i], rsi_obvDF[i]])

        orders = pd.DataFrame(orders, index=pricesDF.index, columns=pricesDF.columns)
        if self.verbose:
            print type(orders)  # it better be a DataFrame!
            print orders
            print prices_all
        orders = orders.diff()
        orders.iloc[0] = 0
        return orders


if __name__ == "__main__":
    print "One does not simply think up a strategy"
