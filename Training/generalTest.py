from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import CategoricalNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import scale
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
import json

def validateMethod(method):
    if not method in ['knn','bayes','forest','svm']:
        print("Invalid learning method")
        sys.exit()

def setupClf(method,param):
    if method == 'knn':
        return KNeighborsClassifier(n_neighbors=param)
    elif method == 'bayes':
        return CategoricalNB(alpha=param)
    elif method == 'forest':
        return RandomForestClassifier(\
            n_estimators=param[0],\
            max_depth=param[1],\
            min_samples_leaf=param[2],\
            ccp_alpha=param[3]\
        )
    elif method == 'svm':
        return SVC(C=param)

def setupParam(method,string):
    if method in ['bayes','svm']:
        return float(string)
    elif method == 'knn':
        return int(string)
    else:
        start = string.split(',')
        for i in range(4):
            start[i] = int(start[i])
        print(start)
        return tuple(start)

def main(method,param,filename,dataFile):
    start = timeit.default_timer()
    nerf = True
    if method == "knn":
        nerf = False
    with open("selectHeaders.json",'r') as f:
        select = json.load(f)
    select.append('satisfied')
    data = readCsv(\
        '../Data/Training/train_'+dataFile+'.csv',\
        nerf=nerf\
    )
    data = selectCols(data,select)
    testData = readCsv(\
        '../Data/Testing/test_'+dataFile+'.csv',\
        nerf=nerf\
    )
    testData = selectCols(testData,select)
    testOriginal = readCsv('../Data/Testing/test.csv')
    print("Reading time: ", timeit.default_timer()-start)

    data = deleteL(data)
    testData = deleteL(testData)

    ids = firstColVector(testOriginal)
    ids = ids[1:]

    start = timeit.default_timer()
    out = separateCol(data,-1)
    data = out[0]
    classes = firstColVector(out[1])
    print("Separation time: ", timeit.default_timer()-start)

    if method == "knn":
        length = len(data)
        combined = data
        combined.extend(testData)
        combined = scale(combined)
        data = combined[:length]
        testData = combined[length:]
    
    start = timeit.default_timer()
    clf = setupClf(method,param)
    clf.fit(data,classes)
    print("Training time: ",timeit.default_timer()-start)

    output = []
    start = timeit.default_timer()
    for i in range(len(testData)):
        output.append([ids[i],clf.predict([data[i]])[0]])

    
    print("Predicting time: ",timeit.default_timer()-start)
    writeCsv("../Output/"+filename+".csv",output)
    
if __name__ == "__main__":
    length = len(sys.argv)
    if length < 4:
        sys.exit()
    method = sys.argv[1]
    validateMethod(method)
    param = setupParam(method,sys.argv[2])
    filename = sys.argv[3]
    if length > 4:
        dataFile = sys.argv[4]
    else:
        dataFile = "processed"
    main(method,param,filename,dataFile)

