# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:03:14 2018

@author: Alon
"""

import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
import random
		  		  		    	 		 		   		 		  
class RTLearner(object):
    def __init__(self, leaf_size = 1, verbose = False):
        self.leaf_size = leaf_size
        self.verbose = verbose
 		  
    def author(self):  		   	  			    		  		  		    	 		 		   		 		  
        return 'aamar32' # replace tb34 with your Georgia Tech username  

    def addEvidence(self,dataX,dataY):
        self.decision_table = self.build_tree(dataX,dataY)
        if self.verbose:
            print "The deciosion table:"
            np.set_printoptions(threshold=np.nan)
            print self.decision_table
        
    def build_tree(self,dataX,dataY):
        dataY = dataY.reshape(len(dataY),)
        if dataY.shape[0] <= self.leaf_size:
            return np.array([[np.nan, dataY.mean(), np.nan, np.nan]])
        elif np.unique(dataY).size == 1:
            return np.array([[np.nan, dataY[0], np.nan, np.nan]])
        else:
            i = random.randint(0,dataX.shape[1]-1)
            split_val = np.median(dataX[:,i])
            leftX = dataX[dataX[:,i]<=split_val]
            leftY = dataY[np.argwhere(dataX[:,i]<=split_val)].flatten()
            rightX = dataX[dataX[:,i]>split_val]
            rightY = dataY[np.argwhere(dataX[:,i]>split_val)].flatten()
            if rightX.shape[0] == 0:
                return np.array([[np.nan, dataY.mean(), np.nan, np.nan]])
            left_tree = self.build_tree(leftX, leftY)
            right_tree = self.build_tree(rightX, rightY)
            root = np.array([[i, split_val, 1, left_tree.shape[0] + 1]])
            return np.concatenate([root, left_tree, right_tree])
        
    def query(self, testX):
        def lookup(data, table):
            i=0
            while i < table.shape[0]:
                if np.isnan(table[i,0]):
                    return table[i,1]
                elif data[int(table[i,0])] <= table[i,1]:
                    i += int(table[i,2])
                else:
                    i += int(table[i,3])
        return np.apply_along_axis(lookup, axis=1, arr=testX, table=self.decision_table)