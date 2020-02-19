from sklearn.neighbors import KNeighborsClassifier
import csv
import numpy as np
import sys
import timeit
import random
import matplotlib.pyplot as plt

def writeFile(path,string):
    with open(path,"w") as f:
        f.write(string)

def zeroPad(string,length):
    copy = string
    while len(copy) < length:
        if len(copy)%2:
            copy = copy + " "
        else:
            copy = " " + copy
    return copy

###############################################################
## A class to hold the enumeration of the string values in a 
## column (as a dictionary) and the integer values in the 
## same column (as a list)
###############################################################

class CompleteEnum:
    enum = {}
    ints = []

    def __init__(self,enum,ints):
        self.enum = enum
        self.ints = ints

class Table:
    pp = 0
    np = 0
    pn = 0
    nn = 0

    def __init__(self):
        pass

    def addObs(self,actual,classified):
        if not actual in [0,1] or not classified in [0,1]:
            print("Invalid argument: ", actual, ",", classifier,")")
            return
        test = (actual,classified)
        if test == (0,0):
            self.nn = self.nn + 1
        elif test == (1,0):
            self.pn = self.pn + 1
        elif test == (0,1):
            self.np = self.np + 1
        elif test == (1,1):
            self.pp = self.pp + 1

    def specificity(self):
        return float(self.nn)/(self.np+self.nn)

    def sensitivity(self):
        return float(self.pp)/(self.pp+self.pn)

    def fScore(self):
        return 2*float(self.nn)/(self.np+self.pn+2*self.nn)

    def accuracy(self):
        return float(self.nn+self.pp)/(self.np+self.nn+self.pp+self.pn)

    def __str__(self):
        length = max([len(str(i)) for i in [self.pp,self.nn,self.np,self.pn]])
        string = "    |Pr +|Pr -|\n"
        string = string + "-"*(6+2*length)+"\n"
        string = string + "Tr +|" + zeroPad(str(self.pp),length) + "|" + zeroPad(str(self.pn),length) + "\n"
        string = string + "-"*(6+2*length)+"\n"
        string = string + "Tr -|" + zeroPad(str(self.np),length) + "|" + zeroPad(str(self.nn),length) + "\n\n"
        string = string + "Sensitivity: " + str(self.sensitivity()) + "\n"
        string = string + "Specificity: " + str(self.specificity()) + "\n"
        string = string + "F: " + str(self.fScore()) + "\n"
        string = string + "Accuracy: " + str(self.accuracy()) + "\n"
        return string

###############################################################
## Converts a string to a number, if possible. If the string 
## is an integer, it will be converted to an integer, if it is 
## a float it will be converted to a float, and if it is 
## neither it will be left as a string.
##
## Parameters: string - the string to be converted
##
## Returns - The input string as a string or number, depending 
##           on what it really is
###############################################################

def convertStoI(string):
    try:
        return int(string)
    except:
        pass
    try: 
        return float(string)
    except:
        return string

###############################################################
## Reads a .csv file, converting strings to numbers where 
## possible
##
## Parameters: path - the filepath of the CSV to be read
##
## Returns - The CSV as a list of lists
###############################################################

def readCsv(path):
    completeCsv = []
    with open(path) as csvFile:
        reader = csv.reader(csvFile, delimiter=',')
        for row in reader:
            completeCsv.append([convertStoI(thing) for thing in row])
    return completeCsv

###############################################################
## Writes a .csv file
##
## Parameters: path - the filepath of the CSV to be written
##             arr - the data to be written to the file
##
## Returns: None
###############################################################

def writeCsv(path,arr):
    with open(path,'w') as csvFile:
        writer = csv.writer(csvFile, delimiter=',')
        for row in arr:
            writer.writerow(row)

###############################################################
## Remove the first row of an array (list of lists)
##
## Parameters: arr - the array to delete the header of
##
## Returns - The original array without the header (first row)
###############################################################

def removeHeader(arr):
    return arr[1:]

###############################################################
## Separates an array by columns into two blocks. All the data 
## will be included in one of the two parts, and none will be 
## duplicated.
##
## Parameters: arr - the array to be split
##             col - the first column to be included in the 
##                   right part
##
## Returns - A list with two elements. The first element is 
##           the left part of the array, and the second element 
##           is the right part of the array
###############################################################

def separateCol(arr,col):
    n = len(arr)
    return [[arr[i][:col] for i in range(n)], [arr[i][col:] for i in range(n)]]

###############################################################
## Deletes the first few columns of an array
##
## Parameters: arr - the array to delete columns from
##             col - the number of rows to be deleted
##
## Returns - A copy of the array with the first "col" columns 
##           deleted
###############################################################

def deleteNCols(arr,col):
    return [arr[i][col:] for i in range(len(arr))]

###############################################################
## Creates a vector from the first column of an array
##
## Parameters: arr - the array to create the vector from
##
## Returns - The elements of the first column of the array as a 
##           list
###############################################################

def firstColVector(arr):
    return [arr[i][0] for i in range(len(arr))]

###############################################################
## Splits a dataset into training and test data by a split of 
## roughly 75-25
##
## Parameters: data - the data to be split
## 
## Returns: A list of lists - [trainData, testData]
###############################################################

def splitSamples(data):
    random.seed(1)
    random.shuffle(data)
    length = len(data)
    cutoff = int(0.2*length*random.random()+0.74*length)
    return [data[:cutoff],data[cutoff:]]

###############################################################
## Checks if there are strings in a specified column of an 
## array
##
## Parameters: arr - the array to be checked for strings
##             col - the column to be checked
##
## Returns - If there are strings in the column, this returns a 
##           list of two lists. The first list is a list of the 
##           unique strings in the column, and the second is a 
##           list of the unique numbers in the list. If there 
##           are no strings in the column, this returns None.
###############################################################

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

###############################################################
## Takes a set of strings and enumerates them, avoiding certain
## values
##
## Parameters: strings - a list of strings to be enumerated
##             ints - a list of disallowed integers
##
## Returns - A dictionary whose keys are the input strings and 
##           whose values are unique integers, none of which 
##           are contained in the list of integers.
###############################################################

def genDict(strings, ints):
    count = 0
    enum = {}
    while len(strings):
        if not count in ints:
            enum[strings[0]] = count
            strings.pop(0)
        count = count + 1
    return enum

###############################################################
## Converts all the strings in a specified column to integers 
## based on a given enumeration of those strings
##
## Parameters: arr - the array to be converted
##             col - the column to be converted 
##             enum - an enumeration of the strings in the column
##
## Returns - None - the conversion is done in place
###############################################################

def adjust(arr,col,enum):
    for i in range(len(arr)):
        if isinstance(arr[i][col],str):
            arr[i][col] = enum[arr[i][col]]

###############################################################
## Iterates through the columns of an array, converting strings 
## to integers and ensuring that the strings in a column are 
## converted to a number that is not already present in that 
## column
##
## Parameters: arr - the array to be converted
##
## Returns - A dictionary in which the keys are the columns
##           that were adjusted and the values are the 
##           enumerations for the columns. We return these so
##           that we use the same enumerations for the test
##           data.
###############################################################

def enumerateStrings(arr):
    completeEnums = {}
    for col in range(len(arr[0])):
        check = checkString(arr,col)
        if not check is None:
            enum = genDict(check[0],check[1])
            completeEnums[col] = CompleteEnum(enum,check[1])
            adjust(arr,col,enum)
    return completeEnums

###############################################################
## Enumerate the test data using the same enumerations as for 
## the training data.
##
## Parameters: arr - the array to be enumerated
##             enums - the enumerations used for the training
##                     data
##
## Returns: None - the conversion is done in place.
###############################################################

def enumerateTestData(arr,enums):
    for col in enums.keys():
        enum = augmentEnumeration(arr,col,completeEnums[col])
        adjust(arr,col,enum)

###############################################################
## Augment the original enumeration for the test data to catch 
## the cases where there are unseen non-numeric entries in the 
## test set
##
## Parameter: arr - the set of test data
##            col - the column to be changed
##            completeEnum - the completeEnum object for the 
##                           column
##
## Returns: a dictionary with the enumeration for all the 
##          string values, which is an augmentation of the 
##          original enumeration
##############################################################

def augmentEnumeration(arr,col,completeEnum):
    check = checkString(arr,col)
    if not check is None:
        strings = [thing for thing in check[0] if not thing in completeEnum.enum.keys()]
        ints = completeEnum.ints + list(completeEnum.enum.values())
        enum = genDict(strings,ints)
        enum.update(completeEnum.enum)
    else:
        enum = completeEnum.enum
    return enum

start = timeit.default_timer()
data = readCsv('../Data/train_numeric.csv')
print("Reading time: ", timeit.default_timer()-start)

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
    for i in range(len(testData)):
        if not i%250-1:
            print("Classifying observation number: ",i)
        tables[k].addObs(testClasses[i],neigh.predict([testData[i]])[0])

    print(str(tables[k]))
    print("Predicting time for k=", k,": ",timeit.default_timer()-start)
    writeFile("../Output/knn_"+str(k),str(tables[k]))

plt.plot(ks,[tables[k].accuracy() for k in ks],'r-')
plt.show()

#test = readCsv('../Data/test_numeric.csv')
