import time
from kNN import kNN
from kNN_skl import kNN_skl


# number of rows
i = 50
k = 5
p = 2

#############################################
# with the homemade method
start_time = time.time()

alg = kNN('mnist', i, k, p)

alg.readData()
#alg.drawElement(alg.dataTest[0].iloc[i].values, 'plots/mnistTrain_{}.pdf'.format(i))
alg.train()
alg.classify()
#print(alg.dataTest[1].head(i))
#print(alg.dataTest[1].head(i)[alg.dataTest[1]['label'] != alg.dataTest[1]['estimate']])
#print(alg.dataTest[1].head(i)[alg.dataTest[1]['label'] != alg.dataTest[1]['estimate']].info())
nerr = len(alg.dataTest[1].head(i)[alg.dataTest[1]['label'] != alg.dataTest[1]['estimate']])

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
#print(alg.dataTest[1].head(i))
#print(alg.dataTest[1].head(i)[alg.dataTest[1]['label'] != alg.dataTest[1]['estimate']])
#print(alg.dataTest[1].head(i)[alg.dataTest[1]['label'] != alg.dataTest[1]['estimate']].info())
nerr = len(alg.dataTest[1].head(i)[alg.dataTest[1]['label'] != alg.dataTest[1]['estimate']])

print("--- SKL alg ---")
print("---   %s accuracy ---" % (1 - nerr/i))
print("---   %s seconds ---" % (time.time() - start_time))
