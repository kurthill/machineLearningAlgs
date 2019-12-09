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
        self.neigh.fit(self.dataTrain[0].values, self.dataTrain[1].values)

    def classify(self):

        y = self.neigh.predict(self.dataTest[0].values)
        self.dataTest[1] = pd.concat([self.dataTest[1], pd.DataFrame(y)], axis=1)
        self.dataTest[1].columns = ['label','estimate']
        
        #print(self.dataTest[1])
