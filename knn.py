from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
import sys
import timeit

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
        if isinstance(thing,str):
            flag = True
            strings.append(thing)
        else:
            nums.append(thing)
    if flag:
        return [np.unique(strings).tolist(),np.unique(nums).tolist()]
    return None 

def genDict(strings, ints):
    count = 0
    enum = {}
    while len(strings):
        if not count in ints:
            enum[strings[0]] = count
            strings.pop(0)
        count = count + 1
    return enum

def adjust(arr,col,enum):
    for i in range(len(arr)):
        if isinstance(arr[i][col],str):
            arr[i][col] = enum[arr[i][col]]

def enumerateStrings(arr):
    for col in range(len(arr[0])):
        check = checkString(arr,col)
        if not check is None:
            enum = genDict(check[0],check[1])
            adjust(data,col,enum)

data = readCsv('../Data/train.csv')
data = removeHeader(data)
data = deleteNCols(data,1)
out = separateCol(data,-1)
data = out[0]
classes = firstColVector(out[1])
enumerateStrings(data)

start = timeit.default_timer()
neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(data,classes)
end = timeit.default_timer()

print("Run time: ",end-start)
