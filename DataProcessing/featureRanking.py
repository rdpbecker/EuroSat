from sklearn.ensemble import ExtraTreesClassifier
import timeit
import sys
sys.path.append("../")
from fileio import *
from dataManip import *

def main():
    start = timeit.default_timer()
    data = readCsv(\
        '../Data/Training/train_numeric.csv'\
    )
    print("Reading time: ", timeit.default_timer()-start)

    headers = data[0]
    dataHeaders = headers[:-1]
    data = data[1:]
    
    start = timeit.default_timer()
    out = separateCol(data,-1)
    data = out[0]
    classes = firstColVector(out[1])
    print("Separation time: ", timeit.default_timer()-start)

    start = timeit.default_timer()
    forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
    forest.fit(data,classes)
    importances = forest.feature_importances_
    print("Training time: ", timeit.default_timer()-start)

    start = timeit.default_timer()
    impDict = {}
    for i in range(len(importances)):
        impDict[dataHeaders[i]] = importances[i]
    featureSort = sorted(dataHeaders,reverse=True,key=lambda x:impDict[x])
    output = [["Feature","Importance (x1000)"]]
    for thing in featureSort:
        print(thing,1000*impDict[thing])
        output.append([thing,1000*impDict[thing]])
    writeCsv("../Data/forestImp.csv",output)
    print("Writing time:", timeit.default_timer()-start)

if __name__ == "__main__":
    main()
