from algorithm import algorithm
import numpy as np
import pandas as pd

"""
Implementation of the support vector machine (linear classifyer)

x vector in R^N classified into M classes via Wx + b -> R^M
W is extended to W' in R^(N+1) x R^M to incorporate b: W'x -> R^M
I.e., x -> (x, 1)

"""

class svm(algorithm):

    def __init__(self, dataName, nsamp=-1):
        self.dataName = dataName
        self.nsamp = nsamp
        self.W # (N+1) weights

    def train(self):
        pass

    def classify(self):
        self.dataTest[1] = pd.DataFrame({
                'label':self.dataTest[1].values,
                'estimate':-1})

        self.dataTest[1]['estimate'] = self.dataTest[1].apply(getNN())


    def loss(self, r, label, delta):
        y = W.dot(r.values) 
        s = np.maximum(0, y[i] - y[label] + delta)
        y[label] = 0
        y.sum()

        
