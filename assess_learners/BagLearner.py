# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 00:15:14 2018

@author: Alon
"""

import numpy as np

class BagLearner(object):
    def __init__(self, learner, kwargs = {}, bags = 20, boost = False, verbose = False):
        self.learners = []
        for i in range(0,bags):
            self.learners.append(learner(**kwargs))
        self.bags = bags
        self.boost = boost
        self.verbose = verbose

    def author(self):
        return 'aamar32' # replace tb34 with your Georgia Tech username  

    def addEvidence(self,dataX,dataY):
        for i in range(0, self.bags):
            n = dataX.shape[0]
            idx = np.random.randint(n, size=n)
            X = dataX[idx]
            y = dataY[idx]
            self.learners[i].addEvidence(X,y)
        
    def query(self, testX):
        self.answer = np.ones([self.bags,testX.shape[0]])
        for i in range(0, self.bags):
            self.answer[i] = self.learners[i].query(testX)
        if self.verbose:
            print self.answer
        return np.mean(self.answer, axis=0)