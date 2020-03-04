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

ks = {\
'knn': [1,2,3,4,6,8,10,15,20,30,40,50,75,100,200,400,700,1000],\
'bayes': [0,0.001,0.01,0.1,1,10,100],\
'forest': [\
#    (10,10,1,0), (10,10,1,0.01),\
#    (10,10,10,0), (10,10,10,0.01),\
#    (10,10,100,0), (10,10,100,0.01),\
#    (10,15,1,0), (10,15,1,0.01),\
#    (10,15,10,0), (10,15,10,0.01),\
#    (10,15,100,0), (10,15,100,0.01),\
#    (10,20,1,0), (10,20,1,0.01),\
#    (10,20,10,0), (10,20,10,0.01),\
#    (10,20,100,0), (10,20,100,0.01),\
#    (50,10,1,0), (50,10,1,0.01),\
#    (50,10,10,0), (50,10,10,0.01),\
#    (50,10,100,0), (50,10,100,0.01),\
#    (50,15,1,0), (50,15,1,0.01),\
#    (50,15,10,0), (50,15,10,0.01),\
#    (50,15,100,0), (50,15,100,0.01),\
#    (50,20,1,0), (50,20,1,0.01),\
#    (50,20,10,0), (50,20,10,0.01),\
#    (50,20,100,0), (50,20,100,0.01),\
    (100,10,1,0), (100,10,1,0.01),\
    (100,10,10,0), (100,10,10,0.01),\
    (100,10,100,0), (100,10,100,0.01),\
    (100,15,1,0), (100,15,1,0.01),\
    (100,15,10,0), (100,15,10,0.01),\
    (100,15,100,0), (100,15,100,0.01),\
    (100,20,1,0), (100,20,1,0.01),\
    (100,20,10,0), (100,20,10,0.01),\
    (100,20,100,0), (100,20,100,0.01),\
],\
'svm': [0.1,0.2,0.35,0.5,0.7,0.85,1]\
}

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

def main(method,suffix):
    start = timeit.default_timer()
    nerf = True
    if method in ["knn","svm"]:
        nerf = False
    with open("selectHeaders.json",'r') as f:
        select = json.load(f)
    select.append('satisfied')
    data = readCsv(\
        '../Data/Training/train_numeric.csv',\
        nerf=nerf\
    )
    data = selectCols(data,select)
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

    if method in ["knn","svm"]:
        trainData = scale(trainData)
        testData = scale(testData)
    
    params = ks[method]
    tables = {}
    for param in params:
        tables[param] = Table()
        print("param=",param)
        start = timeit.default_timer()
        clf = setupClf(method,param)
        clf.fit(trainData,trainClasses)
        print("Training time for param=",param,": ",timeit.default_timer()-start)
    
        start = timeit.default_timer()
        helpers.createTable(tables[param],clf,testData,testClasses)

        print(str(tables[param]))
        print("Predicting time for param=", param,": ",timeit.default_timer()-start)
        writeFile("../../Output/"+method+"_"+str(param)+suffix,str(tables[param]))
    
    if not method == 'forest':
        plt.plot(params,[tables[param].accuracy() for param in params],'r-')
        plt.show()
    
if __name__ == "__main__":
    length = len(sys.argv)
    if length > 1:
        method = sys.argv[1]
        validateMethod(method)
    else:
        sys.exit()
    if length > 2:
        suffix = sys.argv[2]
    else:
        suffix = ""
    main(method,suffix)
