"""  		   	  			    		  		  		    	 		 		   		 		  
Test a learner.  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  
  		   	  			    		  		  		    	 		 		   		 		  
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
"""  		   	  			    		  		  		    	 		 		   		 		  

import util
import numpy as np		  
import math  		   	  			    		  		  		    	 		 		   		 		  
import LinRegLearner as lrl  
import DTLearner as dt		   	  			
import RTLearner as rt   
import BagLearner as bl 	
import InsaneLearner as it	  		  		    	 		 		   		 		  
import sys  		
import matplotlib.pyplot as plt    	  			    		  		  		    	 		 		   		 		  
from time import time



def get_data_dict(dataName, portion = False, shuffle = False):
    with util.get_learner_data_file(dataName) as f:  		   	  			    		  		  		    	 		 		   		 		  
        data = np.genfromtxt(f,delimiter=',')  		   	  			    		  		  		    	 		 		   		 		  
        # Skip the date column and header row if we're working on Istanbul data  		   	  			    		  		  		    	 		 		   		 		  
        if dataName == 'Istanbul.csv':  		   	  			    		  		  		    	 		 		   		 		  
            data = data[1:,1:]
    # compute how much of the data is training and testing  
    if shuffle:		   
        np.random.shuffle(data)
    
    myData = {}	  			    		  		  		    	 		 		   		 		  
    train_rows = int(0.6* data.shape[0]) 
    if portion:
        data = data[:int(portion*data.shape[0])]
          	
    train_rows = int(0.6* data.shape[0])		  		  		    	 		 		   		 		  
    myData['trainX'] = data[:train_rows,0:-1]  		   	  			    		  		  		    	 		 		   		 		  
    myData['trainY'] = data[:train_rows,-1]  		   	  			    		  		  		    	 		 		   		 		  
    myData['testX'] = data[train_rows:,0:-1]  		   	  			    		  		  		    	 		 		   		 		  
    myData['testY'] = data[train_rows:,-1]
    return myData

def run_clf(data, name, learner, kwargs = {}, verbose=False):
    trainX, trainY, testX, testY = data['trainX'], data['trainY'], data['testX'], data['testY']  		   	  			    		  		  		    	 		 		   		 		  
    learner = learner(**kwargs) # create a LinRegLearner  		   	  			    		  		  		    	 		 		   		 		  
    learner.addEvidence(trainX, trainY) # train it  		   	  			    		  		  		    	 		 		   		 		  		   	  			    		  		  		    	 		 		   		 		  
    predY = learner.query(trainX) # get the predictions  		   	  			    		  		  		    	 		 		   		 		  
    rmse_in = math.sqrt(((trainY - predY) ** 2).sum()/trainY.shape[0])  		   	  			    		  		  		    	 		 		   		 		  	   	  			    		  		  		    	 		 		   		 		  
    c_in = np.corrcoef(predY, y=trainY)  		   	  			    		  		  		    	 		 		   		 		   		   	  			    		  		  		    	 		 		   		 		  
    predY = learner.query(testX) # get the predictions  		   	  			    		  		  		    	 		 		   		 		  
    rmse_out = math.sqrt(((testY - predY) ** 2).sum()/testY.shape[0])  		   	  			    		  		  		    	 		 		   		 		   	  			    		  		  		    	 		 		   		 		  
    c_out = np.corrcoef(predY, y=testY)  		   	  			    		  		  		    	 		 		   		 		  
    
    if verbose:
        print "###############",name,"###############" 	
        print learner.author()
        print  		   	  			    		  		  		    	 		 		   		 		  
        print "In sample results"  		   	  			    		  		  		    	 		 		   		 		  
        print "RMSE: ", rmse_in
        print "corr: ", c_in[0,1]
        print  		   	  			    		  		  		    	 		 		   		 		  
        print "Out of sample results"  		   	  			    		  		  		    	 		 		   		 		  
        print "RMSE: ", rmse_out
        print "corr: ", c_out[0,1]
    return rmse_in, rmse_out, c_in, c_out, learner
		
def plot_rmse(mean_num, data, name, learner, kwargs = {}, save=False):
    leaf_num = np.arange(1,60)
    rmse_IN = np.ones([mean_num, leaf_num.shape[0]])
    rmse_OUT = np.ones([mean_num, leaf_num.shape[0]])
    for j in range(mean_num):
        rmse_in = np.ones([leaf_num.shape[0]])
        rmse_out = np.ones([leaf_num.shape[0]])
        for i in range(leaf_num.shape[0]):
            if name == "DTLearner":
                kwargs['leaf_size'] = i
            elif name == "BagLearner":
                kwargs['kwargs']['leaf_size'] = i
            rmse_in[i], rmse_out[i], _, _, _ = run_clf(data, name, learner, kwargs = kwargs)
        rmse_IN[j] = rmse_in
        rmse_OUT[j] = rmse_out
    plt.title(name + ": Leaf Size Overfitting")
    plt.xlabel("leaf_size")
    plt.ylabel("RMSE")
    plt.ylim(0,0.01)
    plt.grid(True)
    plt.plot(leaf_num, rmse_IN.mean(axis=0), label="In Sample", color="darkorange", lw=2)
    plt.plot(leaf_num, rmse_OUT.mean(axis=0), label="Out Sample", color="navy", lw=2)
    plt.legend(loc="best")
    if save:
        plt.savefig("rmse - " + name + ".png", dpi=300)	
    plt.show()    		
    
def plot_time_dt_vs_rt(mean_num, dataName, save=False):
    data_num = np.linspace(0.1,1,10)
    time_DT = np.ones([mean_num, data_num.shape[0]])
    time_RT = np.ones([mean_num, data_num.shape[0]])
    for j in range(mean_num):
        time_dt = np.ones([data_num.shape[0]])
        time_rt = np.ones([data_num.shape[0]])
        for i in range(data_num.shape[0]):
            data = get_data_dict(dataName, portion = data_num[i],shuffle = True)
            t0 = time()
            run_clf(data,"DTLearner", dt.DTLearner, kwargs = {"leaf_size":11})
            time_dt[i] = time() - t0
            t0 = time()
            run_clf(data,"RTLearner", rt.RTLearner, kwargs = {"leaf_size":11})
            time_rt[i] = time() - t0
        time_DT[j] = time_dt
        time_RT[j] = time_rt
    plt.title("DT vs. RT - Times")
    plt.xlabel("% Samples")
    plt.ylabel("Time")
    plt.grid(True)
    plt.plot(data_num, time_DT.mean(axis=0), label="DTLearner", color="darkorange", lw=2)
    plt.plot(data_num, time_RT.mean(axis=0), label="RTLearner", color="navy", lw=2)
    plt.legend(loc="best")
    if save:
        plt.savefig("dtVSrt_time.png", dpi=300)	
    plt.show()
    
def plot_time_dt_vs_rt2(mean_num, dataName, save=False):
    leaf_num = np.arange(1,60)
    size_DT = np.ones([mean_num, leaf_num.shape[0]])
    size_RT = np.ones([mean_num, leaf_num.shape[0]])
    for j in range(mean_num):
        size_dt = np.ones([leaf_num.shape[0]])
        size_rt = np.ones([leaf_num.shape[0]])
        for i in range(leaf_num.shape[0]):
            data = get_data_dict(dataName,shuffle = True)
            t0 = time()
            run_clf(data,"DTLearner", dt.DTLearner, kwargs = {"leaf_size":i})
            size_dt[i] = time() - t0
            t0 = time()
            run_clf(data,"RTLearner", rt.RTLearner, kwargs = {"leaf_size":i})
            size_rt[i] = time() - t0
        size_DT[j] = size_dt
        size_RT[j] = size_rt
    plt.title("DT vs. RT - Times")
    plt.xlabel("leaf_size")
    plt.ylabel("Time")
    plt.grid(True)
    plt.plot(leaf_num, size_DT.mean(axis=0), label="DTLearner", color="darkorange", lw=2)
    plt.plot(leaf_num, size_RT.mean(axis=0), label="RTLearner", color="navy", lw=2)
    print 
    plt.legend(loc="best")
    if save:
        plt.savefig("dtVSrt_time2.png", dpi=300)	
    plt.show()
    
def plot_size_dt_vs_rt(mean_num, dataName, save=False):
    data_num = np.linspace(0.1,1,10)
    size_DT = np.ones([mean_num, data_num.shape[0]])
    size_RT = np.ones([mean_num, data_num.shape[0]])
    for j in range(mean_num):
        size_dt = np.ones([data_num.shape[0]])
        size_rt = np.ones([data_num.shape[0]])
        for i in range(data_num.shape[0]):
            data = get_data_dict(dataName, portion = data_num[i],shuffle = True)
            _, _, _, _, test1 = run_clf(data,"DTLearner", dt.DTLearner, kwargs = {"leaf_size":11})
            size_dt[i] = np.size(test1.decision_table)
            _, _, _, _, test2 = run_clf(data,"RTLearner", rt.RTLearner, kwargs = {"leaf_size":11})
            size_rt[i] = np.size(test2.decision_table)
        size_DT[j] = size_dt
        size_RT[j] = size_rt
    plt.title("DT vs. RT - Size")
    plt.xlabel("% Samples")
    plt.ylabel("Size")
    plt.grid(True)
    plt.plot(data_num, size_DT.mean(axis=0), label="DTLearner", color="darkorange", lw=2)
    plt.plot(data_num, size_RT.mean(axis=0), label="RTLearner", color="navy", lw=2)
    print 
    plt.legend(loc="best")
    if save:
        plt.savefig("dtVSrt_space.png", dpi=300)	
    plt.show()
    
    
def plot_size_dt_vs_rt2(mean_num, dataName, save=False):
    leaf_num = np.arange(1,60)
    size_DT = np.ones([mean_num, leaf_num.shape[0]])
    size_RT = np.ones([mean_num, leaf_num.shape[0]])
    for j in range(mean_num):
        size_dt = np.ones([leaf_num.shape[0]])
        size_rt = np.ones([leaf_num.shape[0]])
        for i in range(leaf_num.shape[0]):
            data = get_data_dict(dataName, shuffle = True)
            _, _, _, _, test1 = run_clf(data,"DTLearner", dt.DTLearner, kwargs = {"leaf_size":i})
            size_dt[i] = np.size(test1.decision_table)
            _, _, _, _, test2 = run_clf(data,"RTLearner", rt.RTLearner, kwargs = {"leaf_size":i})
            size_rt[i] = np.size(test2.decision_table)
        size_DT[j] = size_dt
        size_RT[j] = size_rt
    plt.title("DT vs. RT - Size")
    plt.xlabel("leaf_size")
    plt.ylabel("Size")
    plt.grid(True)
    plt.plot(leaf_num, size_DT.mean(axis=0), label="DTLearner", color="darkorange", lw=2)
    plt.plot(leaf_num, size_RT.mean(axis=0), label="RTLearner", color="navy", lw=2)
    print 
    plt.legend(loc="best")
    if save:
        plt.savefig("dtVSrt_space2.png", dpi=300)	
    plt.show()
	  		  		    	 		 		   		 		  
if __name__=="__main__":  		   	  			    		  		  		    	 		 		   		 		  
    verbose = False
    if len(sys.argv) > 3:  		   	  			    		  		  		    	 		 		   		 		  
        print "Usage: python testlearner.py <filename> [-v]"  		   	  			    		  		  		    	 		 		   		 		  
        sys.exit(1)  	
    elif len(sys.argv) == 3 and sys.argv[2] == "-v":
        verbose = True
    dataName = sys.argv[1]	  		    	 		 		   		 		  
    myData = get_data_dict(dataName, shuffle = False)
    
    
    run_clf(myData, "LinRegLearner", lrl.LinRegLearner, verbose=verbose)
    run_clf(myData, "DTLearner", dt.DTLearner, kwargs={'leaf_size':1}, verbose=verbose)
    run_clf(myData, "RTLearner", rt.RTLearner, kwargs={'leaf_size':1}, verbose=verbose)
    run_clf(myData, "BagLearner", bl.BagLearner, kwargs={"learner" : rt.RTLearner, "kwargs" : {"leaf_size":1}, "bags" : 20}, verbose=verbose)		   	  			    		  		  		    	 		 		   		 		   		   	
    run_clf(myData, "InsaneLearner", it.InsaneLearner, verbose=verbose)	
    
    plot_rmse(10, get_data_dict(dataName, shuffle = True), "DTLearner", dt.DTLearner, save=False) 
    plot_rmse(4, get_data_dict(dataName, shuffle = True), "BagLearner", bl.BagLearner, kwargs={"learner" : rt.RTLearner, "kwargs" : {"leaf_size":1}}, save=False) 	

    plot_time_dt_vs_rt(10, dataName, save=False)        
    plot_time_dt_vs_rt2(10, dataName, save=False)
    plot_size_dt_vs_rt(10, dataName, save=False)        
    plot_size_dt_vs_rt2(10, dataName, save=False)
    
    

    
    
    
    
    
    
    