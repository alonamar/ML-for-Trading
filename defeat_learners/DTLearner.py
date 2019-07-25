"""  		   	  			    		  		  		    	 		 		   		 		  
A simple wrapper for linear regression.  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
Note, this is NOT a correct DTLearner; Replace with your own implementation.  		   	  			    		  		  		    	 		 		   		 		  
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
GT User ID: aamr32 (replace with your User ID)  		   	  			    		  		  		    	 		 		   		 		  
GT ID: 903339940 (replace with your GT ID)  		   	  			    		  		  		    	 		 		   		 		  
"""  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
import numpy as np  		   	  			    		  		  		    	 		 		   		 		  
  	
		  		  		    	 		 		   		 		  
class DTLearner(object):
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
            coeff = abs(np.corrcoef(dataX.T, dataY)[-1,:-1])
            i = np.ma.array(coeff, mask=np.isnan(coeff)).argmax()
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