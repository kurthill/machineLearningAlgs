import time
from svm import svm
import pandas as pd
from svm_skl import svm_skl


# number of rows
i = None
k = 5
p = 2

"""
"""
#############################################
# with the homemade method
start_time = time.time()

alg = svm('mnist', i)

alg.readData()
alg.drawElement(alg.dataTest.iloc[30,1:].values, 'plots/mnistTest_{}.pdf'.format(i))

#alg.train(10, 100)
alg.train(1000, 50)
alg.classify()

#print(alg.dataTest.head(20))

nerr = len(alg.dataTest[alg.dataTest['label'] != alg.dataTest['estimate']])

print("--- Homemade alg ---")
print("---   %s accuracy ---" % (1 - nerr/len(alg.dataTest)))
print("---   %s seconds ---" % (time.time() - start_time))

for i in range(10):
    alg.drawElement(alg.W[i,:-1], 'plots/W_{}.pdf'.format(i))


#############################################
# with the skl method
start_time = time.time()

alg = svm_skl('mnist', i)

alg.readData()
#alg.drawElement(alg.dataTest[0].iloc[i].values, 'plots/mnistTrain_{}.pdf'.format(i))
alg.train()
alg.classify()

nerr = len(alg.dataTest[alg.dataTest['label'] != alg.dataTest['estimate']])

print("--- SKL alg ---")
print("---   %s accuracy ---" % (1 - nerr/len(alg.dataTest)))
print("---   %s seconds ---" % (time.time() - start_time))
