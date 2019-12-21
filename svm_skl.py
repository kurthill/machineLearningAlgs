from algorithm import algorithm
import numpy as np
import pandas as pd
#from sklearn.svm import LinearSVC
from sklearn import linear_model


class svm_skl(algorithm):

    def __init__(self, dataName='mnist', nsamp=-1, lambd=1):
        self.dataName = dataName
        self.nsamp = nsamp
        #self.clf = LinearSVC(C=lambd, loss='hinge')
        self.clf = linear_model.SGDClassifier(alpha=1/lambd, 
                                              learning_rate='optimal', 
                                              loss='hinge')

    def train(self):
        self.clf.fit(self.dataTrain.iloc[:, 1:].values, 
                       self.dataTrain.iloc[:, 0].values)

    def classify(self):

        y = self.clf.predict(self.dataTest.iloc[:, 1:].values)
        self.dataTest = pd.concat(
                [self.dataTest, 
                 pd.DataFrame(y).rename({0: 'estimate'}, axis=1)], axis=1)
        
        #print(self.dataTest[1])
