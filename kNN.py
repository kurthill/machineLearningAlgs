from algorithm import algorithm
import numpy as np
import pandas as pd

class kNN(algorithm):

    def __init__(self, dataName, nsamp=-1, k=5, norm=1):
        self.dataName = dataName
        self.nsamp = nsamp
        self.k = k
        self.norm = norm

    def train(self):
        pass

    def classify(self):
        self.dataTest['estimate'] = self.dataTest.iloc[:, 1:].apply(self.getNN, axis=1)


    # for row r, return the label of the most frequent of the k nearest neighbors
    def getNN(self, r):
        sdf = (((self.dataTrain.iloc[:, 1:] - r)**self.norm)
                .sum(axis=1)
                .sort_values())

        #print(sdf)

        l = [self.dataTrain['label'].loc[sdf.index[i]] for i in range(self.k)]

        #print(l)

        # count the frequency of NNs
        freq = {}
        for j in l:
            freq[l.count(j)] = j
            
        # the most frequent value wins
        max = 0
        for j in freq:
            if j > max:
                max = freq[j]

        return max
