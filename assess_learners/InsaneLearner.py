# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 02:04:58 2018

@author: Alon
"""

import numpy as np 
import BagLearner as bl 
import LinRegLearner as lrl 
class InsaneLearner(object):
    def __init__(self, verbose = False):
        self.verbose = verbose
        self.learner = bl.BagLearner(learner = bl.BagLearner, kwargs = {"learner":lrl.LinRegLearner, "kwargs": {"verbose" : verbose}, "verbose": verbose},
                                     verbose = verbose)
    def author(self):
        return 'aamar32'
    def addEvidence(self,dataX,dataY):
        self.learner.addEvidence(dataX,dataY)    
    def query(self, testX):
        return self.learner.query(testX)
