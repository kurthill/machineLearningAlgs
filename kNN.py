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
        self.dataTest[1] = pd.DataFrame({
                'label':self.dataTest[1].values,
                'estimate':-1})

        self.dataTest[1]['estimate'] = self.dataTest[1].apply(getNN())


    # for row r, return the indicies of the k nearest neighbors as a 
    # list ordered from closest to furthest
    def getNN(self, r):
        sdf = ((self.dataTrain[0] - r)
                .apply(lambda x: pow(abs(x), self.norm))
                .sum(axis=1)
                .sort_values())

        l = [self.dataTrain[1].loc[sdf.index[i]] for i in range(self.k)]

        # count the frequency of NNs
        freq = {}
        for j in l:
            freq[l.count(j)] = j
            
        # the most frequent value wins
        max = 0
        for j in freq:
            if j > max:
                max = j

        return max
