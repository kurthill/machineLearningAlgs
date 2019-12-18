import time
from svm import svm
#from kNN_skl import kNN_skl


# number of rows
i = -1#20
k = 5
p = 2

"""
"""
#############################################
# with the homemade method
start_time = time.time()

alg = svm('mnist', i)

alg.readData()
#alg.drawElement(alg.dataTest[0].iloc[i].values, 'plots/mnistTrain_{}.pdf'.format(i))
alg.train()
alg.classify()

print(alg.dataTest.head(20))

nerr = len(alg.dataTest[alg.dataTest['label'] != alg.dataTest['estimate']])

print("--- Homemade alg ---")
print("---   %s accuracy ---" % (1 - nerr/len(alg.dataTest)))
print("---   %s seconds ---" % (time.time() - start_time))


#############################################
# with the skl method
#start_time = time.time()
#
#
#print("--- SKL alg ---")
#print("---   %s accuracy ---" % (1 - nerr/i))
#print("---   %s seconds ---" % (time.time() - start_time))
