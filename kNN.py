from algorithm import algorithm
import numpy as np
import pandas as pd

class kNN(algorithm):

    def __init__(self, dataName, k=5, norm=1):
        self.dataName = dataName
        self.k = k
        self.norm = norm

    def train(self, dat, label):
        self.dat = dat
        self.label = label

    def classify(self):
        self.dataTest[1] = pd.DataFrame({
                'label':self.dataTest[1].values,
                'estimate':-1})

        #for i in range(len(self.dataTest[0])):
        for i in range(0, 1000):

            # get list of NNs
            l = self.getNN(self.dataTest[0].iloc[i])
            print(l)

            # count the frequency of NNs
            freq = {}
            for j in l:
                freq[l.count(j)] = j
            print(freq)
            
            # the most frequent value wins
            max = 0
            for j in freq:
                if j > max:
                    max = j

            print('The estimate at index {} is {}'.format(i, freq[max]))
            self.dataTest[1].loc[i, 'estimate'] = int(freq[max])
            print(self.dataTest[1].iloc[i])


    # for row r, return the indicies of the k nearest neighbors as a 
    # list ordered from closest to furthest
    def getNN(self, r):
        sdf = ((self.dataTrain[0] - r)
                .apply(lambda x: pow(abs(x), self.norm))
                .sum(axis=1)
                .sort_values())

        #return [sdf.index[i] for i in range(self.k)]
        #return [sdf.iloc[i] for i in range(self.k)]
        return [self.dataTrain[1].loc[sdf.index[i]] for i in range(self.k)]
