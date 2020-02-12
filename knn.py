from sklearn.neighbors import KNeighborsClassifier
import csv

def readCsv(path):
    completeCsv = []
    with open(path) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            completeCsv.append(row)
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

data = readCsv('../Data/train.csv')
data = removeHeader(data)
data = deleteNCols(data,1)
out = separateCol(data,-1)
data = out[0]
classes = firstColVector(out[1])
print(data[0])

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(data,classes)
