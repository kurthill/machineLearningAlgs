import numpy as np
import pandas as pd

class algorithm:

    def __init__(self, dataName, nsamp=-1):
        self.dataName = dataName
        self.nsamp = nsamp
        self.dataTrain = [None, None] # data, labels
        self.dataTest = [None, None] # data, labels

    def readData(self):
        if self.dataName == 'mnist':
            df = pd.read_csv('data/mnist_train.csv', header=None) 
            self.dataTrain = [df.iloc[0: , 1:],
                              df.iloc[0: , 0]]
            #self.dataTrain = [df.drop(0, axis=1), df[0]]
            df = pd.read_csv('data/mnist_test.csv', header=None) 
            self.dataTest = [df.iloc[0:self.nsamp, 1:],
                             df.iloc[0:self.nsamp, 0]]


    def drawElement(self, ele, name):
        import matplotlib as mpl
        mpl.get_backend()
        import matplotlib.pylab as plt

        if self.dataName == 'mnist':
            plt.imshow(ele.reshape(28, 28), cmap='gray')

        plt.savefig(name)

    #def trainAlg():

    #def classify():
