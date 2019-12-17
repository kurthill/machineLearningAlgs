import time
from kNN import kNN
from kNN_skl import kNN_skl


# number of rows
i = 5
k = 5
p = 2

"""
"""
#############################################
# with the homemade method
start_time = time.time()

alg = kNN('mnist', i, k, p)

alg.readData()
#alg.drawElement(alg.dataTest[0].iloc[i].values, 'plots/mnistTrain_{}.pdf'.format(i))
alg.train()
alg.classify()
#print(alg.dataTest.head(i))
#print(alg.dataTest.head(i)[alg.dataTest['label'] != alg.dataTest['estimate']])
#print(alg.dataTest.head(i)[alg.dataTest['label'] != alg.dataTest['estimate']].info())
nerr = len(alg.dataTest.head(i)[alg.dataTest['label'] != alg.dataTest['estimate']])

print("--- Homemade alg ---")
print("---   %s accuracy ---" % (1 - nerr/i))
print("---   %s seconds ---" % (time.time() - start_time))


#############################################
# with the skl method
start_time = time.time()

#alg = kNN('mnist', 20, 2)
alg = kNN_skl('mnist', i, k, p)

alg.readData()
#alg.drawElement(alg.dataTest[0].iloc[i].values, 'plots/mnistTrain_{}.pdf'.format(i))
alg.train()
alg.classify()
nerr = len(alg.dataTest.head(i)[alg.dataTest['label'] != alg.dataTest['estimate']])


print("--- SKL alg ---")
print("---   %s accuracy ---" % (1 - nerr/i))
print("---   %s seconds ---" % (time.time() - start_time))
