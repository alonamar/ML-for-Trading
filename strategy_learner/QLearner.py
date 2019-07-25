"""  		   	  			    		  		  		    	 		 		   		 		  
Template for implementing QLearner  (c) 2015 Tucker Balch  		   	  			    		  		  		    	 		 		   		 		  

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

import numpy as np
import random as rand


class QLearner(object):

    def __init__(self,
                 num_states=100,
                 num_actions=4,
                 alpha=0.2,
                 gamma=0.9,
                 rar=0.5,
                 radr=0.99,
                 dyna=0,
                 verbose=False):

        self.num_states = num_states
        self.dyna = dyna
        self.radr = radr
        self.rar = rar
        self.gamma = gamma
        self.alpha = alpha
        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.q = np.zeros((num_states, num_actions))
        self.t_c = np.full((num_states, num_actions, num_states), 0.00001)
        self.t = np.full((num_states, num_actions, num_states), 1./num_states)
        self.r = np.zeros((num_states, num_actions))
        self.memory = []

    @staticmethod
    def author():
        return 'aamar32'

    def update_q(self, s, a, s_prime, r):
        self.q[s, a] = (1 - self.alpha) * self.q[s, a] \
                       + self.alpha * (r + self.gamma * self.q[s_prime, self.q[s_prime].argmax()])

    def run_dyna(self):
        for i in range(self.dyna):
            s = rand.randint(0, self.num_states - 1)
            a = rand.randint(0, self.num_actions - 1)
            s_prime = np.random.choice(self.num_states, 1, p=self.t[s, a, :])
            r = self.r[s, a]
            self.update_q(s, a, s_prime, r)

    def mrs(self):
        for i in range(self.dyna):
            exp = self.memory[rand.randint(0, len(self.memory)-1)]
            self.update_q(exp[0], exp[1], exp[2], exp[3])

    def querysetstate(self, s):
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the state without updating the Q-table  		   	  			    		  		  		    	 		 		   		 		  
        @param s: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """
        self.s = s
        action = self.q[s].argmax()
        if self.verbose:
            print "s =", s, "a =", action
        return action

    def query(self, s_prime, r):
        """  		   	  			    		  		  		    	 		 		   		 		  
        @summary: Update the Q table and return an action  		   	  			    		  		  		    	 		 		   		 		  
        @param s_prime: The new state  		   	  			    		  		  		    	 		 		   		 		  
        @param r: The ne state  		   	  			    		  		  		    	 		 		   		 		  
        @returns: The selected action  		   	  			    		  		  		    	 		 		   		 		  
        """
        #self.t_c[self.s, self.a, s_prime] += 1
        #self.t[self.s, self.a, :] = self.t_c[self.s, self.a, :] / self.t_c[self.s, self.a, :].sum()
        #self.r[self.s, self.a] = (1 - self.alpha) * self.r[self.s, self.a] + self.alpha * r

        self.update_q(self.s, self.a, s_prime, r)
        self.memory.append([self.s, self.a, s_prime, r])
        self.s = s_prime

        action = self.q[s_prime].argmax()
        if rand.uniform(0.0, 1.0) <= self.rar:
            action = rand.randint(0, self.num_actions - 1)
            self.rar *= self.radr
        self.a = action

        self.mrs()
        #self.run_dyna()

        if self.verbose:
            print "s =", s_prime, "a =", action, "r =", r
        return action


if __name__ == "__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
