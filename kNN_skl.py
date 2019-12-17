from algorithm import algorithm
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

class kNN_skl(algorithm):

    def __init__(self, dataName, nsamp=-1, k=5, norm=1):
        self.dataName = dataName
        self.nsamp = nsamp
        self.norm = norm
        self.neigh = KNeighborsClassifier(n_neighbors=k, p=norm)

    def train(self):
        self.neigh.fit(self.dataTrain.iloc[:, 1:].values, self.dataTrain.iloc[:, 0].values)

    def classify(self):

        y = self.neigh.predict(self.dataTest.iloc[:, 1:].values)
        self.dataTest = pd.concat(
                [self.dataTest, pd.DataFrame(y).rename({0: 'estimate'}, axis=1)], axis=1)
        
        #print(self.dataTest[1])
