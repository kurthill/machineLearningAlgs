from kNN import kNN

#i = 8

alg = kNN('mnist', 20, 2)
alg.readData()
#alg.drawElement(alg.dataTest[0].iloc[i].values, 'plots/mnistTrain_{}.pdf'.format(i))
alg.classify()
print(alg.dataTest[1].head(1000))
print(alg.dataTest[1].head(1000)[alg.dataTest[1]['label'] != alg.dataTest[1]['estimate']])
print(alg.dataTest[1].head(1000)[alg.dataTest[1]['label'] != alg.dataTest[1]['estimate']].info())
