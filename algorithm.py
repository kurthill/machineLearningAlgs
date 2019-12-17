import numpy as np
import pandas as pd

class algorithm:

    def __init__(self, dataName, nsamp=-1):
        self.dataName = dataName
        self.nsamp = nsamp
        self.dataTrain = None
        self.dataTest = None

    def readData(self):
        if self.dataName == 'mnist':
            self.dataTrain = (pd.read_csv('data/mnist_train.csv', header=None) 
                                .rename({0: 'label'}, axis=1))

            self.dataTest = (pd.read_csv('data/mnist_test.csv', header=None) 
                               .iloc[0: self.nsamp]
                               .rename({0: 'label'}, axis=1))


    def drawElement(self, ele, name):
        import matplotlib as mpl
        mpl.get_backend()
        import matplotlib.pylab as plt

        if self.dataName == 'mnist':
            plt.imshow(ele.reshape(28, 28), cmap='gray')

        plt.savefig(name)

    #def trainAlg():

    #def classify():
