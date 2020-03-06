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
import Voters
import numpy as np

dataChoice = {'bayes': 0, 'forest': 0, 'knn': 1, 'svm': 1}

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
        for i in range(3):
            start[i] = int(start[i])
        start[3] = float(start[3])
        print(start)
        return tuple(start)

def main(methods,params):
    start = timeit.default_timer()
    with open("selectHeaders.json",'r') as f:
        select = json.load(f)
    select.append('satisfied')
    data = [\
        readCsv(\
            '../Data/Training/train_numeric.csv',\
            nerf=True\
        ),\
        readCsv(\
            '../Data/Training/train_numeric.csv',\
            nerf=False\
        )\
    ]
    print("Reading time: ", timeit.default_timer()-start)

    for i in range(2):
        data[i] = selectCols(data[i],select)
        headers = data[i][0]
        data[i] = data[i][1:]

    # Scale data[1] together, but don't scale the ids or the
    # classes
    col1 = [data[1][i][0] for i in range(len(data[1]))]
    classes = [data[1][i][-1] for i in range(len(data[1]))]
    data[1] = [data[1][i][1:-1] for i in range(len(data[1]))]
    data[1] = scale(data[1])
    data[1] = data[1].tolist()
    for i in range(len(data[1])):
        data[1][i].insert(0,col1[i])
        data[1][i].append(classes[i])
    
    start = timeit.default_timer()
    splitout = [[],[]]
    splitout[0] = splitSamples(data[0])
    ids = [splitout[0][0][i][0] for i in range(len(splitout[0][0]))]
    splitout[1] = selectObsById(data[1],ids)
    trainDatas = [[],[]]
    testDatas = [[],[]]
    trainClasseses = [[],[]]
    testClasseses = [[],[]]
    for i in range(2):
        trainDatas[i] = splitout[i][0]
        testDatas[i] = splitout[i][1]
        
        out = separateCol(trainDatas[i],-1)
        trainDatas[i] = out[0]
        trainDatas[i] = deleteNCols(trainDatas[i],1)
        trainClasseses[i] = firstColVector(out[1])
        
        out = separateCol(testDatas[i],-1)
        testDatas[i] = out[0]
        testIds = [testDatas[i][j][0] for j in range(len(testDatas[i]))]
        testDatas[i] = deleteNCols(testDatas[i],1)
        testClasseses[i] = firstColVector(out[1])
        print("Separation time: ", timeit.default_timer()-start)

    arrs = []
    for i in range(len(methods)):
        print('Starting method', methods[i])
        table = Table()
        arr = [["ids","Predicted"]]
        # Choose data
        trainData = trainDatas[dataChoice[methods[i]]]
        testData = testDatas[dataChoice[methods[i]]]
        trainClasses = trainClasseses[dataChoice[methods[i]]]
        testClasses = testClasseses[dataChoice[methods[i]]]

        start = timeit.default_timer()
        clf = setupClf(methods[i],params[i])
        clf.fit(trainData,trainClasses)
        print("Training time",methods[i],": ",timeit.default_timer()-start)

        start = timeit.default_timer()
        helpers.createTableWithArr(table,arr,clf,testData,testClasses,testIds)
        arrs.append(arr)

        print(str(table))
        print("Predicting time",methods[i],": ",timeit.default_timer()-start)

    voters = Voters.Voters()
    table = Table()
    for arr in arrs:
        voters.addVoter(arr)
    for i in range(len(testIds)):
        table.addObs(testClasses[i],voters.vote(testIds[i]))
    print(str(table))
    writeFile("../../Output/ensemble/"+methods[0]+"_"+str(params[0]),str(table))
    
if __name__ == "__main__":
    length = len(sys.argv)
    if length < 3:
        sys.exit()
    methods = []
    params = []
    for i in range(int(length/2)):
        method = sys.argv[2*i+1]
        validateMethod(method)
        param = setupParam(method,sys.argv[2*i+2])
        methods.append(method)
        params.append(param)
    main(methods,params)
