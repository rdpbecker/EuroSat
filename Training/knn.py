from sklearn.neighbors import KNeighborsClassifier
import sys
sys.path.append("../Classes")
sys.path.append("../DataProcessing")
sys.path.append("../")
from dataManip import *
from fileio import *
from Table import Table
import timeit
import matplotlib.pyplot as plt

def main():
    start = timeit.default_timer()
    data = readCsv('../Data/Training/train_processed.csv')
    print("Reading time: ", timeit.default_timer()-start)
    
    data = deleteL(data)

    start = timeit.default_timer()
    out = splitSamples(data)
    trainData = out[0]
    testData = out[1]
    
    out = separateCol(trainData,-1)
    trainData = out[0]
    trainClasses = firstColVector(out[1])
    
    out = separateCol(testData,-1)
    testData = out[0]
    testClasses = firstColVector(out[1])
    print("Separation time: ", timeit.default_timer()-start)
    
    ks = [1,2,3,4,6,8,10,15,20,30,40,50,75,100,200,400,700,1000]
    tables = {}
    for k in [1,2,3,4,6,8,10,15,20,30,40,50,75,100,200,400,700,1000]:
        tables[k] = Table()
        print("k=",k)
        start = timeit.default_timer()
        neigh = KNeighborsClassifier(n_neighbors=k)
        neigh.fit(trainData,trainClasses)
        print("Training time for k=",k,": ",timeit.default_timer()-start)
    
        start = timeit.default_timer()
        createTable(tables[k],neigh,testData,testClasses)
    
        print(str(tables[k]))
        print("Predicting time for k=", k,": ",timeit.default_timer()-start)
        writeFile("../../Output/knn2_"+str(k),str(tables[k]))
    
    plt.plot(ks,[tables[k].accuracy() for k in ks],'r-')
    plt.show()
    
if __name__ == "__main__":
    main()
