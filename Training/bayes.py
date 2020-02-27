from sklearn.naive_bayes import CategoricalNB
import sys
sys.path.append("../Classes")
sys.path.append("../DataProcessing")
sys.path.append("../")
from dataManip import *
from fileio import *
from Table import Table
import timeit
import matplotlib.pyplot as plt
import clfHelpers as helpers

def main():
    start = timeit.default_timer()
    data = readCsv('../Data/Training/train_processed.csv',nerf=True)
    print("Reading time: ", timeit.default_timer()-start)

    headers = data[0]
    data = data[1:]
    data = deleteNCols(data,1)
    
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
    
    ks = [0.001,0.01,0.1,1,10,100]
    tables = {}
    for k in ks:
        tables[k] = Table()
        print("k=",k)
        start = timeit.default_timer()
        bayes = CategoricalNB()
        bayes.fit(trainData,trainClasses)
        print("Training time for k=",k,": ",timeit.default_timer()-start)
    
        start = timeit.default_timer()
        helpers.createTable(tables[k],bayes,testData,testClasses)

        print(str(tables[k]))
        print("Predicting time for k=", k,": ",timeit.default_timer()-start)
        writeFile("../../Output/bayes_"+str(k),str(tables[k]))
    
    plt.plot(ks,[tables[k].accuracy() for k in ks],'r-')
    plt.show()
    
if __name__ == "__main__":
    main()
