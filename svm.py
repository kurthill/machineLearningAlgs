from algorithm import algorithm
import numpy as np
import pandas as pd
import random

"""
Implementation of the support vector machine (linear classifyer) with
soft margin

x vector in R^N classified into M classes via Wx + b -> R^M
W is extended to W' in R^(N+1) x R^M to incorporate b: W'x -> R^M
I.e., x -> (x, 1)

Weights determined with stocastic gradient descent 
"""

class svm(algorithm):

    def __init__(self, dataName='mnist', nsamp=-1, delta=1, lambd=1):
        self.dataName = dataName
        self.nsamp = nsamp
        self.delta = delta
        self.lambd = lambd
        self.W = self.__init_W()


    def __init_W(self):
        W = None # (N+1) weights
        if self.dataName == 'mnist':
            #W = np.empty((10, 785)) # 10 x (N+1) weights 
            W = np.random.randn(10, 785) * 0.01 # small random weights
            #W = np.full( (785, 10), 1) # (N+1) x 10 weights
        return W


    def train(self, nit = 100, nbatch = 1):
        self.dataTrain['b'] = 1
        self.dataTest['b'] = 1

        # implement pegasos algorithm on minibatches
        # random minibatch
        for t in range(1, nit+1):
            eta = 1/(self.lambd*t) # decaying step size
            j = random.randint(0, len(self.dataTrain) - nbatch)
            # initialize grad and average over minibatch
            gW = np.full((10, 785), 0)
            for i in range(j,j + nbatch):
                gW += self.grad(self.dataTrain.iloc[i], self.W)

            # average minibatch
            gW = gW/nbatch
            # updata W = W - stepsize ( grad + regularization )
            self.W = self.W - eta*(gW + self.lambd*self.W)

        #print(gW)
        #print(self.W)


    def classify(self):
        self.dataTest['estimate'] = self.dataTest.apply(self.classRow, axis=1)
        #df = self.dataTest.copy()
        #df['estimate'] = df.apply(self.classRow, axis=1)
        #print(df)
        #return df


    def classRow(self, r):
        return np.argmax(self.W.dot(r[1:].values))


    # loss function to calculate 'loss' 
    #   takes a data vector and the weights matrix
    #   returns a loss ndarray of dim N_categories
    def __loss_fun(self, r, W):

        y = W.dot(r[1:].values)
        return  y - y[r['label']] + self.delta


    # loss function to be applied to the training set via the 'apply' method
    #   takes a data vector and the weights matrix
    #   returns the scaler sum of losses
    def loss(self, r, W):

        l = np.maximum(0, self.__loss_func(r, W))
        l[r['lable']] = 0

        return l.sum()


    # analytic gradiant w/ for loops
    #   takes a data vector and the weights matrix
    #   returns the gradient of the W.r as a ndarray of dim W
    def grad(self, r, W):

        gW = np.full((10, 785), 0)

        l = self.__loss_fun(r, W)

        count = 0
        for i in range(len(l)):
            if (l[i] > 0) and (i != r['label']):
                count += 1
                gW[i] = r[1:].values

        gW[r['label']] = -1 * count * r[1:].values

        return gW

