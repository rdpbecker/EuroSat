from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
import sys

def convertStoI(string):
    try:
        return int(string)
    except:
        pass
    try: 
        return float(string)
    except:
        return string

def readCsv(path):
    completeCsv = []
    with open(path) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            completeCsv.append([convertStoI(thing) for thing in row])
    return completeCsv

def removeHeader(arr):
    return arr[1:]

def separateCol(arr,col):
    n = len(arr)
    return [[arr[i][:col] for i in range(n)], [arr[i][col:] for i in range(n)]]

def deleteNCols(arr,col):
    return [arr[i][col:] for i in range(len(arr))]

def firstColVector(arr):
    return [arr[i][0] for i in range(len(arr))]

def checkString(arr,col):
    strings = []
    nums = []
    flag = False
    for i in range(len(arr)):
        thing = arr[i][col]
        if thing and isinstance(thing,str):
            flag = True
            strings.append(thing)
        else:
            nums.append(thing)
    if flag:
        return [np.unique(strings),np.unique(nums)]
    return None 

data = readCsv('../Data/train.csv')
data = removeHeader(data)
data = deleteNCols(data,1)
out = separateCol(data,-1)
data = out[0]
classes = firstColVector(out[1])
for col in range(len(data[0])):
    check = checkString(data,col)
    if not check is None:
        print(col,check[0],check[1])
sys.exit()

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(data,classes)
